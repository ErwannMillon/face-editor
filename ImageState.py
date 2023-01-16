import numpy as np
import gc
import os
import imageio
import glob
import uuid
from animation import clear_img_dir
from backend import ImagePromptEditor, log
import torch
import torchvision
import wandb
from edit import blend_paths
from img_processing import custom_to_pil
from PIL import Image

num = 0


class PromptTransformHistory:
    def __init__(self, iterations) -> None:
        self.iterations = iterations
        self.transforms = []


class ImageState:
    def __init__(self, vqgan, prompt_optimizer: ImagePromptEditor) -> None:
        self.vqgan = vqgan
        self.device = vqgan.device
        self.blend_latent = None
        self.quant = True
        self.path1 = None
        self.path2 = None
        self.img_dir = "./img_history"
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)
        self.transform_history = []
        self.attn_mask = None
        self.prompt_optim = prompt_optimizer
        self._load_vectors()
        self.init_transforms()

    def _load_vectors(self):
        self.lip_vector = torch.load(
            "./latent_vectors/lipvector.pt", map_location=self.device
        )
        self.blue_eyes_vector = torch.load(
            "./latent_vectors/2blue_eyes.pt", map_location=self.device
        )
        self.asian_vector = torch.load(
            "./latent_vectors/asian10.pt", map_location=self.device
        )

    def create_gif(self, total_duration, extend_frames, gif_name="face_edit.gif"):
        images = []
        folder = self.img_dir
        paths = glob.glob(folder + "/*")
        frame_duration = total_duration / len(paths)
        print(len(paths), "frame dur", frame_duration)
        durations = [frame_duration] * len(paths)
        if extend_frames:
            durations[0] = 1.5
            durations[-1] = 3
        for file_name in os.listdir(folder):
            if file_name.endswith(".png"):
                file_path = os.path.join(folder, file_name)
                images.append(imageio.imread(file_path))
        imageio.mimsave(gif_name, images, duration=durations)
        return gif_name

    def init_transforms(self):
        self.blue_eyes = torch.zeros_like(self.lip_vector)
        self.lip_size = torch.zeros_like(self.lip_vector)
        self.asian_transform = torch.zeros_like(self.lip_vector)
        self.current_prompt_transforms = [torch.zeros_like(self.lip_vector)]

    def clear_transforms(self):
        self.init_transforms()
        clear_img_dir("./img_history")
        return self._render_all_transformations()

    def _latent_to_pil(self, latent):
        current_im = self.vqgan.decode(latent.to(self.device))[0]
        return custom_to_pil(current_im)

    def _get_mask(self, img, mask=None):
        if img and "mask" in img and img["mask"] is not None:
            attn_mask = torchvision.transforms.ToTensor()(img["mask"])
            attn_mask = torch.ceil(attn_mask[0].to(self.device))
            print("mask set successfully")
        else:
            attn_mask = mask
        return attn_mask

    def set_mask(self, img):
        self.attn_mask = self._get_mask(img)
        x = self.attn_mask.clone()
        x = x.detach().cpu()
        x = torch.clamp(x, -1.0, 1.0)
        x = (x + 1.0) / 2.0
        x = x.numpy()
        x = (255 * x).astype(np.uint8)
        x = Image.fromarray(x, "L")
        return x

    @torch.inference_mode()
    def _render_all_transformations(self, return_twice=True):
        global num
        current_vector_transforms = (
            self.blue_eyes,
            self.lip_size,
            self.asian_transform,
            sum(self.current_prompt_transforms),
        )
        new_latent = self.blend_latent + sum(current_vector_transforms)
        if self.quant:
            new_latent, _, _ = self.vqgan.quantize(new_latent.to(self.device))
        image = self._latent_to_pil(new_latent)
        image.save(f"{self.img_dir}/img_{num:06}.png")
        num += 1
        return (image, image) if return_twice else image

    def apply_rb_vector(self, weight):
        self.blue_eyes = weight * self.blue_eyes_vector
        return self._render_all_transformations()

    def apply_lip_vector(self, weight):
        self.lip_size = weight * self.lip_vector
        return self._render_all_transformations()

    def update_quant(self, val):
        self.quant = val
        return self._render_all_transformations()

    def apply_asian_vector(self, weight):
        self.asian_transform = weight * self.asian_vector
        return self._render_all_transformations()

    def update_images(self, path1, path2, blend_weight):
        if path1 is None and path2 is None:
            return None

        # Duplicate paths if one is empty
        if path1 is None:
            path1 = path2
        if path2 is None:
            path2 = path1

        self.path1, self.path2 = path1, path2
        if self.img_dir:
            clear_img_dir(self.img_dir)
        return self.blend(blend_weight)

    @torch.inference_mode()
    def blend(self, weight):
        _, latent = blend_paths(
            self.vqgan,
            self.path1,
            self.path2,
            weight=weight,
            show=False,
            device=self.device,
        )
        self.blend_latent = latent
        return self._render_all_transformations()

    @torch.inference_mode()
    def rewind(self, index):
        if not self.transform_history:
            print("No history")
            return self._render_all_transformations()
        prompt_transform = self.transform_history[-1]
        latent_index = int(index / 100 * (prompt_transform.iterations - 1))
        print(latent_index)
        self.current_prompt_transforms[-1] = prompt_transform.transforms[
            latent_index
        ].to(self.device)
        return self._render_all_transformations()

    def _init_logging(lr, iterations, lpips_weight, positive_prompts, negative_prompts):
        wandb.init(reinit=True, project="face-editor")
        wandb.config.update({"Positive Prompts": positive_prompts})
        wandb.config.update({"Negative Prompts": negative_prompts})
        wandb.config.update(
            dict(lr=lr, iterations=iterations, lpips_weight=lpips_weight)
        )

    def apply_prompts(
        self,
        positive_prompts,
        negative_prompts,
        lr,
        iterations,
        lpips_weight,
        reconstruction_steps,
    ):
        if log:
            self._init_logging(
                lr, iterations, lpips_weight, positive_prompts, negative_prompts
            )
        transform_log = PromptTransformHistory(iterations + reconstruction_steps)
        transform_log.transforms.append(
            torch.zeros_like(self.blend_latent, requires_grad=False)
        )
        self.current_prompt_transforms.append(
            torch.zeros_like(self.blend_latent, requires_grad=False)
        )
        positive_prompts = [prompt.strip() for prompt in positive_prompts.split("|")]
        negative_prompts = [prompt.strip() for prompt in negative_prompts.split("|")]
        self.prompt_optim.set_params(
            lr,
            iterations,
            lpips_weight,
            attn_mask=self.attn_mask,
            reconstruction_steps=reconstruction_steps,
        )

        for i, transform in enumerate(
            self.prompt_optim.optimize(
                self.blend_latent, positive_prompts, negative_prompts
            )
        ):
            transform_log.transforms.append(transform.detach().cpu())
            self.current_prompt_transforms[-1] = transform
            with torch.inference_mode():
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
