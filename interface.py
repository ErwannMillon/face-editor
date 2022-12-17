import os
import sys

import wandb

sys.path.append("taming-transformers")
import functools

import gradio as gr
from transformers import CLIPModel, CLIPProcessor

import edit
# import importlib
# importlib.reload(edit)
from app_backend import ImagePromptOptimizer, ImageState, ProcessorGradientFlow
from loaders import load_default

device = "cuda"
vqgan = load_default(device)
vqgan.eval()
processor = ProcessorGradientFlow(device=device)
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip.to(device)
promptoptim = ImagePromptOptimizer(vqgan, clip, processor, quantize=True)
state = ImageState(vqgan, promptoptim)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            hair_red_blue = gr.Slider(
                label="Blue Eyes",
                minimum=-.8,
                maximum=3.,
                value=0,
                step=0.1,
            )
            hair_green_purple = gr.Slider(
                label="hair green<->purple ",
                minimum=-.8,
                maximum=.8,
                value=0,
                step=0.1,
            )
            lip_size = gr.Slider(
                label="Lip Size",
                minimum=-1.9,
                value=0,
                maximum=1.9,
                step=0.1,
            )
            blend_weight = gr.Slider(
                label="Blend faces: 0 is base image, 1 is the second img",
                minimum=-0.,
                value=0,
                maximum=1.,
                step=0.1,
            )
            requantize = gr.Checkbox(
                label="Requantize Latents (necessary using text prompts)",
                value=True,
            )
            gender_weight = gr.Slider(
                label="Asian",
                minimum=-2.,
                value=0,
                maximum=2.,
                step=0.07,
            )
            with gr.Row():
                with gr.Column(scale=1):
                    base_img = gr.Image(label="base Image", type="filepath")
                    blend_img = gr.Image(label="image for face blending (optional)", type="filepath")
        with gr.Column(scale=1):
            out = gr.Image(tool="sketch", shape=(400, 400))
            rewind = gr.Slider(value=100,
                                label="Rewind back through a prompt transform: Use this to scroll through the iterations of your prompt transformation.",
                                minimum=0,
                                maximum=100)
        with gr.Column(scale=1):
            positive_prompts = gr.Textbox(label="Positive prompts",
                                            value="a picture of a woman with a very big nose | a picture of a woman with a large wide nose | a woman with an extremely prominent nose")
            negative_prompts = gr.Textbox(label="Negative prompts",
                                            value="a picture of a person with a tiny nose | a picture of a person with a very thin nose")
            iterations = gr.Slider(minimum=10,
                                    maximum=300,
                                    step=1,
                                    value=40,
                                    label="optimization iterations",)
            learning_rate = gr.Slider(minimum=1e-3,
                                    maximum=6e-1,
                                    value=1e-2,
                                    label="learning rate")
           
            lpips_weight = gr.Slider(minimum=0,
                                    maximum=50,
                                    value=1,
                                    label="Perceptual similarity (high to preserve identity for transformations where the person's identity should not change, recommended when masking)")
            reconstruction_steps = gr.Slider(minimum=0,
                                    maximum=50,
                                    value=15,
                                    step=1,
                                    label="Steps to run optimizing only masked perceptual loss. This helps to reconstruct the original identity for prompts that tend to modify the identity too much")
            apply_prompts = gr.Button(value="Apply Prompts")
    gender_weight.change(state.apply_gender_vector, inputs=[gender_weight], outputs=out)
    requantize.change(state.update_requant, inputs=[requantize], outputs=out)
    lip_size.change(state.apply_lip_vector, inputs=[lip_size], outputs=out)
    hair_green_purple.change(state.apply_gp_vector, inputs=[hair_green_purple], outputs=out)
    hair_red_blue.change(state.apply_rb_vector, inputs=[hair_red_blue], outputs=out)
    blend_weight.change(state.blend, inputs=[base_img, blend_img, blend_weight], outputs=out)
    base_img.change(state.blend, inputs=[base_img, base_img, blend_weight], outputs=out)
    blend_img.change(state.blend, inputs=[base_img, blend_img, blend_weight], outputs=out)
    apply_prompts.click(state.apply_prompts, inputs=[positive_prompts, negative_prompts, learning_rate, iterations, out, lpips_weight, reconstruction_steps], outputs=out)
    rewind.change(state.rewind, inputs=[rewind], outputs=out)
if __name__ == "__main__":
    demo.queue()
    demo.launch(debug=True, inbrowser=True)
