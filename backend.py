import matplotlib.pyplot as plt
import torch
import torchvision
import wandb
from torch import nn
from tqdm import tqdm
from transformers import CLIPProcessor
from img_processing import get_pil, loop_post_process


global log
log = False

class ProcessorGradientFlow:
    """
    This wraps the huggingface CLIP processor to allow backprop through the image processing step.
    The original processor forces conversion to numpy then PIL images, which is faster for image processing but breaks gradient flow.
    """

    def __init__(self, device="cuda") -> None:
        self.device = device
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.image_mean = [0.48145466, 0.4578275, 0.40821073]
        self.image_std = [0.26862954, 0.26130258, 0.27577711]
        self.normalize = torchvision.transforms.Normalize(
            self.image_mean, self.image_std
        )
        self.resize = torchvision.transforms.Resize(224)
        self.center_crop = torchvision.transforms.CenterCrop(224)

    def preprocess_img(self, images):
        images = self.center_crop(images)
        images = self.resize(images)
        images = self.center_crop(images)
        images = self.normalize(images)
        return images

    def __call__(self, images=[], **kwargs):
        processed_inputs = self.processor(**kwargs)
        processed_inputs["pixel_values"] = self.preprocess_img(images)
        processed_inputs = {
            key: value.to(self.device) for (key, value) in processed_inputs.items()
        }
        return processed_inputs


class ImagePromptEditor(nn.Module):
    def __init__(
        self,
        vqgan,
        clip,
        clip_preprocessor,
        lpips_fn,
        iterations=100,
        lr=0.01,
        save_vector=True,
        return_val="vector",
        quantize=True,
        make_grid=False,
        lpips_weight=6.2,
    ) -> None:

        super().__init__()
        self.latent = None
        self.device = vqgan.device
        vqgan.eval()
        self.vqgan = vqgan
        self.clip = clip
        self.iterations = iterations
        self.lr = lr
        self.clip_preprocessor = clip_preprocessor
        self.make_grid = make_grid
        self.return_val = return_val
        self.quantize = quantize
        self.lpips_weight = lpips_weight
        self.perceptual_loss = lpips_fn

    def set_latent(self, latent):
        self.latent = latent.detach().to(self.device)

    def set_params(self, lr, iterations, lpips_weight, reconstruction_steps, attn_mask):
        self.attn_mask = attn_mask
        self.iterations = iterations
        self.lr = lr
        self.lpips_weight = lpips_weight
        self.reconstruction_steps = reconstruction_steps

    def forward(self, vector):
        base_latent = self.latent.detach().requires_grad_()
        trans_latent = base_latent + vector
        if self.quantize:
            z_q, *_ = self.vqgan.quantize(trans_latent)
        else:
            z_q = trans_latent
        dec = self.vqgan.decode(z_q)
        return dec

    def _get_clip_similarity(self, prompts, image, weights=None):
        if isinstance(prompts, str):
            prompts = [prompts]
        elif not isinstance(prompts, list):
            raise TypeError("Provide prompts as string or list of strings")
        clip_inputs = self.clip_preprocessor(
            text=prompts, images=image, return_tensors="pt", padding=True
        )
        clip_outputs = self.clip(**clip_inputs)
        similarity_logits = clip_outputs.logits_per_image
        if weights:
            similarity_logits *= weights
        return similarity_logits.sum()

    def _get_CLIP_loss(self, pos_prompts, neg_prompts, image):
        pos_logits = self._get_clip_similarity(pos_prompts, image)
        if neg_prompts:
            neg_logits = self._get_clip_similarity(neg_prompts, image)
        else:
            neg_logits = torch.tensor([1], device=self.device)
        loss = -torch.log(pos_logits) + torch.log(neg_logits)
        return loss

    def _apply_mask(self, grad):
        newgrad = grad
        if self.attn_mask is not None:
            newgrad = grad * (self.attn_mask)
        return newgrad

    def _apply_inverse_mask(self, grad):
        newgrad = grad
        if self.attn_mask is not None:
            newgrad = grad * ((self.attn_mask - 1) * -1)
        return newgrad

    def _get_next_inputs(self, transformed_img):
        processed_img = loop_post_process(transformed_img)
        processed_img.retain_grad()

        lpips_input = processed_img.clone()
        lpips_input.register_hook(self._apply_inverse_mask)
        lpips_input.retain_grad()

        clip_input = processed_img.clone()
        clip_input.register_hook(self._apply_mask)
        clip_input.retain_grad()

        return (processed_img, lpips_input, clip_input)

    def _optimize_CLIP_LPIPS(self, optim, original_img, vector, pos_prompts, neg_prompts):
        for i in (range(self.iterations)):
            optim.zero_grad()
            transformed_img = self(vector)
            processed_img, lpips_input, clip_input = self._get_next_inputs(
                transformed_img
            )
            # with torch.autocast("cuda"):
            clip_loss = self._get_CLIP_loss(pos_prompts, neg_prompts, clip_input)
            print("CLIP loss", clip_loss)
            perceptual_loss = (
                self.perceptual_loss(lpips_input, original_img.clone())
                * self.lpips_weight
            )
            print("LPIPS loss: ", perceptual_loss)
            print("Sum Loss", perceptual_loss + clip_loss)
            if log:
                wandb.log({"Perceptual Loss": perceptual_loss})
                wandb.log({"CLIP Loss": clip_loss})
            
            # These gradients will be masked if attn_mask has been set
            clip_loss.backward(retain_graph=True)
            perceptual_loss.backward(retain_graph=True)

            optim.step()
            yield vector

    def _optimize_LPIPS(self, vector, original_img, optim):
        for i in range(self.reconstruction_steps):
            optim.zero_grad()
            transformed_img = self(vector)
            processed_img = loop_post_process(transformed_img)
            processed_img.retain_grad()

            lpips_input = processed_img.clone()
            lpips_input.register_hook(self._apply_inverse_mask)
            lpips_input.retain_grad()
            with torch.autocast("cuda"):
                perceptual_loss = (
                    self.perceptual_loss(lpips_input, original_img.clone())
                    * self.lpips_weight
                )
            if log:
                wandb.log({"Perceptual Loss": perceptual_loss})
            print("LPIPS loss: ", perceptual_loss)
            perceptual_loss.backward(retain_graph=True)
            optim.step()
            yield vector

    def optimize(self, latent, pos_prompts, neg_prompts):
        self.set_latent(latent)
        transformed_img = self(
            torch.zeros_like(self.latent, requires_grad=True, device=self.device)
        )
        original_img = loop_post_process(transformed_img)
        vector = torch.randn_like(self.latent, requires_grad=True, device=self.device)
        optim = torch.optim.Adam([vector], lr=self.lr)

        for transform in self._optimize_CLIP_LPIPS(optim, original_img, vector, pos_prompts, neg_prompts):
            yield transform

        print("Running LPIPS optim only")
        for transform in self._optimize_LPIPS(vector, original_img, optim):
            yield transform

        yield vector if self.return_val == "vector" else self.latent + vector
