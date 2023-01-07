import os
import sys

from img_processing import custom_to_pil, preprocess, preprocess_vqgan

sys.path.append("taming-transformers")
import glob

import gradio as gr
import matplotlib.pyplot as plt
import PIL
import taming
import torch

from loaders import load_config
from utils import get_device


def get_embedding(model, path=None, img=None, device="cpu"):
    assert path or img, "Input either path or tensor"
    if img is not None:
        raise NotImplementedError
    x = preprocess(PIL.Image.open(path), target_image_size=256).to(device)
    x_processed = preprocess_vqgan(x)
    z, _, [_, _, indices] = model.encode(x_processed)
    return z

    
def blend_paths(model, path1, path2, quantize=False, weight=0.5, show=True, device="cuda"):
    x = preprocess(PIL.Image.open(path1), target_image_size=256).to(device)
    y = preprocess(PIL.Image.open(path2), target_image_size=256).to(device)
    x_latent, y_latent = get_embedding(model, path=path1, device=device), get_embedding(model, path=path2, device=device)
    z = torch.lerp(x_latent, y_latent, weight)
    if quantize:
        z = model.quantize(z)[0]
    decoded = model.decode(z)[0]
    if show:
        plt.figure(figsize=(10, 20))
        plt.subplot(1, 3, 1)
        plt.imshow(x.cpu().permute(0, 2, 3, 1)[0])
        plt.subplot(1, 3, 2)
        plt.imshow(custom_to_pil(decoded))
        plt.subplot(1, 3, 3)
        plt.imshow(y.cpu().permute(0, 2, 3, 1)[0])
        plt.show()
    return custom_to_pil(decoded), z

if __name__ == "__main__":
    device = get_device()
    ckpt_path = "logs/2021-04-23T18-11-19_celebahq_transformer/checkpoints/last.ckpt"
    conf_path = "./unwrapped.yaml"
    config = load_config(conf_path, display=False)
    model = taming.models.vqgan.VQModel(**config.model.params)
    sd = torch.load("./vqgan_only.pt", map_location="mps")
    model.load_state_dict(sd, strict=True)
    model.to(device)
    blend_paths(model, "./test_data/face.jpeg", "./test_data/face2.jpeg", quantize=False, weight=.5)
    plt.show()