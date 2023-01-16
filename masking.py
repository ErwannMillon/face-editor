import os
import sys

import matplotlib.pyplot as plt
import torch
from backend import ImagePromptEditor, ImageState, ProcessorGradientFlow
from loaders import load_default
from transformers import CLIPModel

if __name__ == "__main__":
    sys.path.append("taming-transformers")
    device = "cuda"

    vqgan = load_default(device)
    vqgan.eval()

    processor = ProcessorGradientFlow(device=device)
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip.to(device)

    promptoptim = ImagePromptEditor(vqgan, clip, processor, quantize=True)
    state = ImageState(vqgan, promptoptim)
    mask = torch.load("eyebrow_mask.pt")
    x = state.blend("./test_data/face.jpeg", "./test_data/face2.jpeg", 0.5)
    plt.imshow(x)
    plt.show()
    state.apply_prompts(
        "a picture of a woman with big eyebrows", "", 0.009, 40, None, mask=mask
    )
    print("done")
