from functools import cache
import importlib

import gradio as gr
import matplotlib.pyplot as plt
import torch
import torchvision
import wandb
from icecream import ic
from torch import nn
from torchvision.transforms.functional import resize
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import lpips
from edit import blend_paths
from img_processing import *
from img_processing import custom_to_pil
from loaders import load_default, load_disc
import glob

def make_animation():
    img_dir = "./img_history"
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    for filename in glob.glob(img_dir+"/*"):
        print(filename)
log=False
# ic.disable()
# ic.enable()
def get_resized_tensor(x):
    if len(x.shape) == 2:
        re = x.unsqueeze(0)
    else: re = x
    re = resize(re, (10, 10))
    return re
class ProcessorGradientFlow():
    """
    This wraps the huggingface CLIP processor to allow backprop through the image processing step.
    The original processor forces conversion to PIL images, which breaks gradient flow. 
    """
    def __init__(self, device="cuda") -> None:
        self.device = device
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
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
        self.disc = load_disc(self.device)
        self.lpips_weight = lpips_weight
        self.perceptual_loss = lpips.LPIPS(net='vgg').to(self.device)
    def disc_loss_fn(self, logits):
        return -torch.mean(logits)
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
            neg_logits = torch.tensor([1], device=self.device)
        loss = -torch.log(pos_logits) + torch.log(neg_logits)
        return loss
    def visualize(self, processed_img):
        if self.make_grid:
            self.index += 1
            plt.subplot(1, 13, self.index)
            plt.imshow(get_pil(processed_img[0]).detach().cpu())
        else:
            plt.imshow(get_pil(processed_img[0]).detach().cpu())
            plt.show()
    def attn_masking(self, grad):
        # print("attnmask 1")
        # print(f"input grad.shape = {grad.shape}")
        # print(f"input grad = {get_resized_tensor(grad)}")
        newgrad = grad
        if self.attn_mask is not None:
            # print("masking mult")
            newgrad = grad * (self.attn_mask)
        # print("output grad, ", get_resized_tensor(newgrad))
        # print("end atn 1")
        return newgrad
    def attn_masking2(self, grad):
        # print("attnmask 2")
        # print(f"input grad.shape = {grad.shape}")
        # print(f"input grad = {get_resized_tensor(grad)}")
        newgrad = grad
        if self.attn_mask is not None:
            # print("masking mult")
            newgrad = grad * ((self.attn_mask - 1) * -1)
        # print("output grad, ", get_resized_tensor(newgrad))
        # print("end atn 2")
        return newgrad

    def optimize(self, latent, pos_prompts, neg_prompts):
        self.set_latent(latent)
        # self.make_grid=True
        transformed_img = self(torch.zeros_like(self.latent, requires_grad=True, device=self.device))
        original_img = loop_post_process(transformed_img)
        vector = torch.randn_like(self.latent, requires_grad=True, device=self.device)
        optim = torch.optim.Adam([vector], lr=self.lr)
        if self.make_grid:
            plt.figure(figsize=(35, 25))
            self.index = 1
        for i in tqdm(range(self.iterations)):
            optim.zero_grad()
            transformed_img = self(vector)
            processed_img = loop_post_process(transformed_img) #* self.attn_mask
            processed_img.retain_grad()
            lpips_input = processed_img.clone()
            lpips_input.register_hook(self.attn_masking2)
            lpips_input.retain_grad()
            clip_clone = processed_img.clone()
            clip_clone.register_hook(self.attn_masking)
            clip_clone.retain_grad()
            with torch.autocast("cuda"):
                clip_loss = self.get_similarity_loss(pos_prompts, neg_prompts, clip_clone)
                print("CLIP loss", clip_loss)
                perceptual_loss = self.perceptual_loss(lpips_input, original_img.clone()) * self.lpips_weight
                print("LPIPS loss: ", perceptual_loss)
                with torch.no_grad():
                    disc_logits = self.disc(transformed_img)
                    disc_loss = self.disc_loss_fn(disc_logits)
                    print(f"disc_loss = {disc_loss}")
                    disc_loss2 = self.disc(processed_img)
            if log:
                wandb.log({"Perceptual Loss": perceptual_loss})
                wandb.log({"Discriminator Loss": disc_loss})
                wandb.log({"CLIP Loss": clip_loss})
            clip_loss.backward(retain_graph=True)
            perceptual_loss.backward(retain_graph=True)
            p2 = processed_img.grad
            print("Sum Loss", perceptual_loss + clip_loss)
            optim.step()
            # if i % self.iterations // 10 == 0: 
                # self.visualize(transformed_img)
            yield vector
        if self.make_grid:
            plt.savefig(f"plot {pos_prompts[0]}.png")
            plt.show()
        print("lpips solo op")
        for i in range(self.reconstruction_steps):
            optim.zero_grad()
            transformed_img = self(vector)
            processed_img = loop_post_process(transformed_img) #* self.attn_mask
            processed_img.retain_grad()
            lpips_input = processed_img.clone()
            lpips_input.register_hook(self.attn_masking2)
            lpips_input.retain_grad()
            with torch.autocast("cuda"):
                perceptual_loss = self.perceptual_loss(lpips_input, original_img.clone()) * self.lpips_weight
                with torch.no_grad():
                    disc_logits = self.disc(transformed_img)
                    disc_loss = self.disc_loss_fn(disc_logits)
                    print(f"disc_loss = {disc_loss}")
                    disc_loss2 = self.disc(processed_img)
            # print(f"disc_loss2 = {disc_loss2}")
            if log:
                wandb.log({"Perceptual Loss": perceptual_loss})
            print("LPIPS loss: ", perceptual_loss)
            perceptual_loss.backward(retain_graph=True)
            optim.step()
            yield vector
        # torch.save(vector, "nose_vector.pt")
        # print("")
        # print("DISC STEPS")
        # print("*************")
        # for i in range(self.reconstruction_steps):
        #     optim.zero_grad()
        #     transformed_img = self(vector)
        #     processed_img = loop_post_process(transformed_img) #* self.attn_mask
        #     disc_logits = self.disc(transformed_img)
        #     disc_loss = self.disc_loss_fn(disc_logits)
        #     print(f"disc_loss = {disc_loss}")
        #     if log:
        #         wandb.log({"Disc Loss": disc_loss})
        #     print("LPIPS loss: ", perceptual_loss)
        #     disc_loss.backward(retain_graph=True)
        #     optim.step()
        #     yield vector
        yield vector if self.return_val == "vector" else self.latent + vector
