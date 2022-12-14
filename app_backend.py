import torch.nn as nn
from img_processing import custom_to_pil
from edit import blend_paths
from img_processing import *
from utils import freeze_module
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import gradio as gr
import importlib
import loaders
import functools
import edit
import importlib
from loaders import load_default
import torchvision
from transformers import CLIPProcessor, CLIPModel
import lpips
class ProcessorGradientFlow():
    def __init__(self, device="cuda") -> None:
        self.device = device
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.image_mean = [0.48145466, 0.4578275, 0.40821073]
        self.image_std = [0.26862954, 0.26130258, 0.27577711]
        self.normalize = torchvision.transforms.Normalize(
            self.image_mean,
            self.image_std
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
        processed_inputs = {key:value.to(self.device) for (key, value) in processed_inputs.items()}
        return processed_inputs

class ImagePromptOptimizer(nn.Module):
    def __init__(self, 
                vqgan, 
                clip,
                clip_preprocessor,
                iterations=100,
                lr = 0.01,
                save_vector=True,
                return_val="vector",
                quantize=True,
                make_grid=False,
                lpips_weight = 6.2) -> None:
                
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
        self.perceptual_loss = lpips.LPIPS(net='vgg').to(self.device)
    def set_latent(self, latent):
        self.latent = latent.detach().to(self.device)
    def set_params(self, lr, iterations, attn_mask, lpips_weight):
        self.attn_mask = attn_mask
        self.iterations = iterations
        self.lr = lr
        self.lpips_weight = lpips_weight
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
        clip_inputs = self.clip_preprocessor(text=prompts,
            images=image, return_tensors="pt", padding=True)
        clip_outputs = self.clip(**clip_inputs)
        similarity_logits = clip_outputs.logits_per_image
        if weights:
            similarity_logits *= weights
        return similarity_logits.sum()
    def get_similarity_loss(self, pos_prompts, neg_prompts, image):
        pos_logits = self._get_clip_similarity(pos_prompts, image)
        if neg_prompts:
            neg_logits = self._get_clip_similarity(neg_prompts, image)
        else:
            neg_logits = torch.tensor([0], device=self.device)
        loss = -torch.log(pos_logits) + torch.log(neg_logits)
        return loss
    def visualize(self, processed_img):
        if self.make_grid:
            self.index += 1
            plt.subplot(1, 9, self.index)
            plt.imshow(get_pil(processed_img[0]).detach().cpu())
        else:
            plt.imshow(get_pil(processed_img[0]).detach().cpu())
            plt.show()
    def attn_masking(self, grad):
        print(f"grad.shape = {grad.shape}")
        if self.attn_mask is not None:
            print("masking mult")
            return grad * self.attn_mask
        return grad

    def optimize(self, latent, pos_prompts, neg_prompts):
        self.set_latent(latent)
        transformed_img = self(torch.zeros_like(self.latent, requires_grad=True, device=self.device))
        original_img = loop_post_process(transformed_img)
        vector = torch.randn_like(self.latent, requires_grad=True, device=self.device)
        optim = torch.optim.Adam([vector], lr=self.lr)
        if self.make_grid:
            plt.figure(figsize=(35, 25))
            self.index = 1
        for i in tqdm(range(self.iterations)):
            transformed_img = self(vector)
            processed_img = loop_post_process(transformed_img)
            # processed_img.register_hook(self.attn_masking)
            # p1 = processed_img.retain_grad().grad
            # print(p1)
            # if i < self.iterations - 2:
              # print("masking2")
              # processed_img *= self.attn_mask
            lpips_input = processed_img.clone()
            lpips_input.register_hook(self.attn_masking)
            perceptual_loss = self.perceptual_loss(lpips_input, original_img.clone()) * self.lpips_weight
            print("perc: ", perceptual_loss)
            print("percweight: ", self.lpips_weight)

            clip_loss = self.get_similarity_loss(pos_prompts, neg_prompts, processed_img)  
            loss = clip_loss + perceptual_loss
            print(loss)
            optim.zero_grad()
            # p2 = processed_img.grad
            # print(p2)
            loss.backward(retain_graph=True)
            # p3 = processed_img.retain_grad().grad
            # print(p3, p3.shape)
            optim.step()
            # return
            # if i % self.iterations // 10 == 0: 
            #     self.visualize(processed_img)
            yield vector
        # if self.make_grid:
            # plt.savefig(f"plot {pos_prompts[0]}.png")
            # plt.show()
        yield vector if self.return_val == "vector" else self.latent + vector

class ImageState:
    def __init__(self, vqgan, prompt_optimizer: ImagePromptOptimizer) -> None:
        self.vqgan = vqgan
        self.device = vqgan.device
        #latentvectors
        self.lip_vector = torch.load("./latent_vectors/lipvector.pt", map_location=self.device)
        self.red_blue_vector = torch.load("./latent_vectors/red_blue.pt", map_location=self.device)
        self.green_purple_vector = torch.load("./latent_vectors/green_purple.pt", map_location=self.device)
        # self.gender_vector = torch.load("./latent_vectors/gender.pt")
        self.asian_vector = torch.load("./latent_vectors/asian10.pt", map_location=self.device)
        #latent transforms
        self.hair_rb = torch.zeros_like(self.lip_vector)
        self.lip_transforms = torch.zeros_like(self.lip_vector)
        self.gender_transforms = torch.zeros_like(self.lip_vector)
        self.prompt_transforms = torch.zeros_like(self.lip_vector) 
        self.hair_gp = torch.zeros_like(self.lip_vector)
        self.blend_latent = None
        self.quant = True
        self.prompt_optim = prompt_optimizer
    def _apply_vector(self, src, vector):
        new_latent = torch.lerp(src, src + vector, 1)
        return new_latent
    def _decode_latent(self, latent):
        current_im = self.vqgan.decode(latent.to(self.device))[0]
        return custom_to_pil(current_im)
    def _render_all_transformations(self):
        self.current_vector_transforms = [self.hair_rb, self.lip_transforms, self.hair_gp, self.gender_transforms, self.prompt_transforms]
        new_latent = self.blend_latent + sum(self.current_vector_transforms).to(self.device)
        if self.quant:
            new_latent, _, _ = self.vqgan.quantize(new_latent.to(self.device))
        return self._decode_latent(new_latent)
    def apply_gp_vector(self, weight):
        self.hair_gp = weight * self.green_purple_vector
        return self._render_all_transformations()
    def apply_rb_vector(self, weight):
        self.hair_rb = weight * self.red_blue_vector
        return self._render_all_transformations()
    def apply_lip_vector(self, weight):
        self.lip_transforms = weight * self.lip_vector
        return self._render_all_transformations()
    def update_requant(self, val):
        print(f"val = {val}")
        self.quant = val
        return self._render_all_transformations()
    def apply_gender_vector(self, weight):
        self.gender_transforms = weight * self.asian_vector
        return self._render_all_transformations()
    def blend(self, path1, path2, weight):
        img, latent = blend_paths(self.vqgan, path1, path2, weight=weight, show=False, device=self.device)
        self.blend_latent = latent.to(self.device)
        return self._render_all_transformations()
    def rescale_mask(self, mask):
        rep = mask.clone()
        rep[mask < 0.03] = -1000000
        rep[mask >= 0.03] = 1
        return rep
    def apply_prompts(self, positive_prompts, negative_prompts, lr, iterations, img, lpips_weight, mask=None):
        # attn_mask = mask
        if img and "mask" in img and img["mask"] is not None:
            attn_mask = torchvision.transforms.ToTensor()(img["mask"])
            attn_mask = torch.ceil(attn_mask[0].to(self.device))
            # torch.save(attn_mask, "lip_mask.pt")
            # attn_mask = self.rescale_mask(attn_mask)
            print(type(attn_mask))
            print(attn_mask.shape)
        else:
            attn_mask = torch.ones_like(img, device=self.device)
        self.prompt_optim.set_params(lr, iterations, attn_mask, lpips_weight)
        for i, transform in enumerate(self.prompt_optim.optimize(self.blend_latent,
                                                positive_prompts,
                                                negative_prompts)):
          
          print(i)
          self.prompt_transforms = transform
          yield self._render_all_transformations()
        # transform = self.prompt_optim.optimize(self.blend_latent,
                                                # positive_prompts,
                                                # negative_prompts)
        # self.prompt_transforms = transform
        # return self._render_all_transformations()
