import os
import sys
sys.path.append("taming-transformers")
import functools
import edit
from app_backend import ProcessorGradientFlow, ImagePromptOptimizer, ImageState
from transformers import CLIPProcessor, CLIPModel
from loaders import load_default
import gradio as gr
import torch
import matplotlib.pyplot as plt
from app_backend import get_resized_tensor
device = "cuda"
vqgan = load_default(device)
vqgan.eval()
processor = ProcessorGradientFlow(device=device)
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip.to(device)
promptoptim = ImagePromptOptimizer(vqgan, clip, processor, quantize=True)
state = ImageState(vqgan, promptoptim)
x = state.blend("./test_data/face.jpeg", "./test_data/face2.jpeg", 0.5)
torch.manual_seed(10)
mask = torch.load("nose_mask.pt").to(device)
from img_processing import custom_to_pil
no_mask=torch.ones_like(mask)
# mask = rescale_mask(mask)
# mask = (mask) * -1
print("mask: ", get_resized_tensor(mask))
pos_prompts = "a picture of a woman with a very big nose"
neg_prompts = "a picture of a person with a small nose"
for i, pic in enumerate(state.apply_prompts(pos_prompts, neg_prompts, 0.03, 60, None, 2, mask=mask)):
  if i %2 == 0:
    plt.imshow(pic)
    plt.show()