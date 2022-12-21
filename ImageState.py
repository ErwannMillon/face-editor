# from align import align_from_path
import gc
import imageio
import glob
import uuid
from animation import clear_img_dir
from backend import ImagePromptOptimizer, log
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
from backend import get_resized_tensor
from edit import blend_paths
from img_processing import *
from img_processing import custom_to_pil
from loaders import load_default
num = 0

class PromptTransformHistory():
    def __init__(self, iterations) -> None:
        self.iterations = iterations
        self.transforms = []

class ImageState:
    def __init__(self, vqgan, prompt_optimizer: ImagePromptOptimizer) -> None:
        # global vqgan
        self.vqgan = vqgan
        self.device = vqgan.device
        self.blend_latent = None
        self.quant = True
        self.path1 = None
        self.path2 = None
        self.transform_history = []
        self.attn_mask = None
        self.prompt_optim = prompt_optimizer
        self.state_id = None
        print(self.state_id)
        self._load_vectors()
        self.init_transforms()
    def _load_vectors(self):
        self.lip_vector = torch.load("./latent_vectors/lipvector.pt", map_location=self.device)
        self.red_blue_vector = torch.load("./latent_vectors/2blue_eyes.pt", map_location=self.device)
        self.green_purple_vector = torch.load("./latent_vectors/nose_vector.pt", map_location=self.device)
        self.asian_vector = torch.load("./latent_vectors/asian10.pt", map_location=self.device)
    def create_gif(self, total_duration, extend_frames, gif_name="face_edit.gif"):
        images = []
        folder = self.state_id
        paths = glob.glob(folder + "/*")
        frame_duration = total_duration / len(paths)
        print(len(paths), "frame dur", frame_duration)
        durations = [frame_duration] * len(paths)
        if extend_frames:
            durations [0] = 1.5
            durations [-1] = 3
        for file_name in os.listdir(folder):
            if file_name.endswith('.png'):
                file_path = os.path.join(folder, file_name)
                images.append(imageio.imread(file_path))
        imageio.mimsave(gif_name, images, duration=durations)
        return gif_name
    def init_transforms(self):
        self.blue_eyes = torch.zeros_like(self.lip_vector)
        self.lip_size = torch.zeros_like(self.lip_vector)
        self.asian_transform = torch.zeros_like(self.lip_vector)
        self.current_prompt_transforms = [torch.zeros_like(self.lip_vector)]
        self.hair_gp = torch.zeros_like(self.lip_vector)
    def clear_transforms(self):
        global num
        self.init_transforms()
        clear_img_dir("./img_history")
        num = 0
        return self._render_all_transformations()
    def _apply_vector(self, src, vector):
        new_latent = torch.lerp(src, src + vector, 1)
        return new_latent
    def _decode_latent_to_pil(self, latent):
        current_im = self.vqgan.decode(latent.to(self.device))[0]
        return custom_to_pil(current_im)
    def _get_mask(self, img, mask=None):
        if img and "mask" in img and img["mask"] is not None:
            attn_mask = torchvision.transforms.ToTensor()(img["mask"])
            attn_mask = torch.ceil(attn_mask[0].to(self.device))
            plt.imshow(attn_mask.detach().cpu(), cmap="Blues")
            plt.show()
            torch.save(attn_mask, "test_mask.pt")
            print("mask set successfully")
            # attn_mask = self.rescale_mask(attn_mask)
            print(type(attn_mask))
            print(attn_mask.shape)
        else:
            attn_mask = mask
            print("mask in apply ", get_resized_tensor(attn_mask), get_resized_tensor(attn_mask).shape)
        return attn_mask
    def set_mask(self, img):
        attn_mask = self._get_mask(img)
        self.attn_mask = attn_mask
            # attn_mask = torch.ones_like(img, device=self.device)
        x = attn_mask.clone()
        x = x.detach().cpu()
        x = torch.clamp(x, -1., 1.)
        x = (x + 1.)/2.
        x = x.numpy()
        x = (255*x).astype(np.uint8)
        x = Image.fromarray(x, "L")
        return x
    @torch.no_grad()
    def _render_all_transformations(self, return_twice=True):
        global num
        # global vqgan
        if self.state_id is None:
            self.state_id = "./img_history/" + str(uuid.uuid4())
            print("redner all", self.state_id)
        current_vector_transforms = (self.blue_eyes, self.lip_size, self.hair_gp, self.asian_transform, sum(self.current_prompt_transforms))
        new_latent = self.blend_latent + sum(current_vector_transforms)
        if self.quant:
            new_latent, _, _ = self.vqgan.quantize(new_latent.to(self.device))
        image = self._decode_latent_to_pil(new_latent)
        img_dir = self.state_id
        if not os.path.exists("img_history"):
            os.mkdir("./img_history")
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        image.save(f"{img_dir}/img_{num:06}.png")
        num += 1
        return (image, image) if return_twice else image
    def apply_gp_vector(self, weight):
        self.hair_gp = weight * self.green_purple_vector
        return self._render_all_transformations()
    def apply_rb_vector(self, weight):
        self.blue_eyes = weight * self.red_blue_vector
        return self._render_all_transformations()
    def apply_lip_vector(self, weight):
        self.lip_size = weight * self.lip_vector
        return self._render_all_transformations()
    def update_requant(self, val):
        print(f"val = {val}")
        self.quant = val
        return self._render_all_transformations()
    def apply_asian_vector(self, weight):
        self.asian_transform = weight * self.asian_vector
        return self._render_all_transformations()
    def update_images(self, path1, path2, blend_weight):
        if path1 is None and path2 is None:
            print("no paths")
            return None
        if path1 == path2:
            print("paths are the same")
            print(path1)
        if path1 is None: path1 = path2
        if path2 is None: path2 = path1
        self.path1, self.path2 = path1, path2
        if self.state_id:
            clear_img_dir(self.state_id)
        return self.blend(blend_weight)
    @torch.no_grad()
    def blend(self, weight):
        _, latent = blend_paths(self.vqgan, self.path1, self.path2, weight=weight, show=False, device=self.device)
        self.blend_latent = latent
        return self._render_all_transformations()
    @torch.no_grad()
    def rewind(self, index):
        if not self.transform_history:
            print("no history")
            return self._render_all_transformations()
        prompt_transform = self.transform_history[-1]
        latent_index = int(index / 100 * (prompt_transform.iterations - 1))
        print(latent_index)
        self.current_prompt_transforms[-1] = prompt_transform.transforms[latent_index].to(self.device)
        return self._render_all_transformations()
    def apply_prompts(self, positive_prompts, negative_prompts, lr, iterations, lpips_weight, reconstruction_steps):
        if self.state_id is None:
            self.state_id = "./img_history/" + str(uuid.uuid4())
        transform_log = PromptTransformHistory(iterations + reconstruction_steps)
        transform_log.transforms.append(torch.zeros_like(self.blend_latent, requires_grad=False))
        self.current_prompt_transforms.append(torch.zeros_like(self.blend_latent, requires_grad=False))
        if log:
            wandb.init(reinit=True, project="face-editor")
            wandb.config.update({"Positive Prompts": positive_prompts})
            wandb.config.update({"Negative Prompts": negative_prompts})
            wandb.config.update(dict(
                lr=lr,
                iterations=iterations,
                lpips_weight=lpips_weight
            ))
        positive_prompts = [prompt.strip() for prompt in positive_prompts.split("|")]
        negative_prompts = [prompt.strip() for prompt in negative_prompts.split("|")]
        self.prompt_optim.set_params(lr, iterations, lpips_weight, attn_mask=self.attn_mask, reconstruction_steps=reconstruction_steps)
        for i, transform in enumerate(self.prompt_optim.optimize(self.blend_latent,
                                                                positive_prompts,
                                                                negative_prompts)):
            transform_log.transforms.append(transform.detach().cpu())
            self.current_prompt_transforms[-1] = transform
            with torch.no_grad():
                image = self._render_all_transformations(return_twice=False)
            if log:
                wandb.log({"image": wandb.Image(image)})
            yield (image, image)
        if log:
            wandb.finish()
        self.attn_mask = None
        self.transform_history.append(transform_log)
        gc.collect()
        torch.cuda.empty_cache()
        # transform = self.prompt_optim.optimize(self.blend_latent,
                                                # positive_prompts,
                                                # negative_prompts)
        # self.prompt_transforms = transform
        # return self._render_all_transformations()