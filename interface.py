import os
import sys
sys.path.append("taming-transformers")
import functools
import edit
# import importlib
# importlib.reload(edit)
from app_backend import ProcessorGradientFlow, ImagePromptOptimizer, ImageState
from transformers import CLIPProcessor, CLIPModel
from loaders import load_default
import gradio as gr
device = "cuda"
vqgan = load_default(device)
vqgan.eval()
processor = ProcessorGradientFlow(device=device)
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip.to(device)
promptoptim = ImagePromptOptimizer(vqgan, clip, processor, quantize=True)
state = ImageState(vqgan, promptoptim)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=2):
            hair_red_blue = gr.Slider(
                label="hair red<->blue ",
                minimum=-.8,
                maximum=.8,
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
                label="lip size",
                minimum=-1.9,
                value=0,
                maximum=1.9,
                step=0.1,
            )
            blend_weight = gr.Slider(
                label="0 is src image, 1 is blend_img",
                minimum=-0.,
                value=0,
                maximum=1.,
                step=0.1,
            )
            requantize = gr.Checkbox(
                label="requantize latents",
                value=True,
            )
            gender_weight = gr.Slider(
                label="gender weight (-1 female, 1 male)",
                minimum=-2.,
                value=0,
                maximum=2.,
                step=0.07,
            )
            with gr.Row():
                with gr.Column(scale=1):
                    base_img = gr.Image(label="base Image", type="filepath")
                    blend_img = gr.Image(label="image for face blending (optional)", type="filepath")
                with gr.Column(scale=2):
                    positive_prompts = gr.Textbox(label="Positive prompts",
                                                    value="a picture of a woman with a very big nose | a picture of a woman with a large wide nose | a woman with an extremely prominent nose")
                    negative_prompts = gr.Textbox(label="Negative prompts",
                                                    value="a picture of a person with a tiny nose | a picture of a person with a very thin nose")
                    iterations = gr.Slider(minimum=10,
                                            maximum=300,
                                            value=40,
                                            label="optimization iterations",)
                    learning_rate = gr.Slider(minimum=1e-3,
                                            maximum=1e-1,
                                            value=1e-2,
                                            label="learning rate")
                    clip_weight = gr.Slider(minimum=0,
                                            maximum=30,
                                            value=1,
                                            label="clip loss weight similarity ecommended when masking)")
                    lpips_weight = gr.Slider(minimum=1,
                                            maximum=30,
                                            value=1,
                                            label="Perceptual similarity (high to preserve identity for transformations where the person's identity should not change, recommended when masking)")
                    apply_prompts = gr.Button(value="Apply Prompts")

        with gr.Column(scale=1):
            out = gr.Image(tool="sketch", shape=(200, 200))
            test = gr.Image(interactive=False, shape=(200, 200))
            i = gr.Button()
    gender_weight.change(state.apply_gender_vector, inputs=[gender_weight], outputs=out)
    requantize.change(state.update_requant, inputs=[requantize], outputs=out)
    lip_size.change(state.apply_lip_vector, inputs=[lip_size], outputs=out)
    hair_green_purple.change(state.apply_gp_vector, inputs=[hair_green_purple], outputs=out)
    hair_red_blue.change(state.apply_rb_vector, inputs=[hair_red_blue], outputs=out)
    blend_weight.change(state.blend, inputs=[base_img, blend_img, blend_weight], outputs=out)
    base_img.change(state.blend, inputs=[base_img, base_img, blend_weight], outputs=out)
    blend_img.change(state.blend, inputs=[base_img, blend_img, blend_weight], outputs=out)
    apply_prompts.click(state.apply_prompts, inputs=[positive_prompts, negative_prompts, learning_rate, iterations, out, lpips_weight, clip_weight], outputs=out)
if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True, debug=True, inbrowser=True)
