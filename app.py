import glob
import os
import sys

import wandb

from configs import set_major_global, set_major_local, set_small_local

sys.path.append("taming-transformers")
import functools

import gradio as gr
from transformers import CLIPModel, CLIPProcessor

import edit
# import importlib
# importlib.reload(edit)
from app_backend import ImagePromptOptimizer, ProcessorGradientFlow
from ImageState import ImageState
from loaders import load_default
from animation import create_gif
from prompts import get_random_prompts

device = "cuda"
vqgan = load_default(device)
vqgan.eval()
processor = ProcessorGradientFlow(device=device)
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip.to(device)
def set_img_from_example(img):
    return state.update_images(img, img, 0)
def get_cleared_mask():
    return gr.Image.update(value=None)
    # mask.clear()
with gr.Blocks(css="styles.css") as demo:
    promptoptim = ImagePromptOptimizer(vqgan, clip, processor, quantize=True)
    state = ImageState(vqgan, promptoptim)
    with gr.Row():
        with gr.Column(scale=1):
            blue_eyes = gr.Slider(
                label="Blue Eyes",
                minimum=-.8,
                maximum=3.,
                value=0,
                step=0.1,
            )
            # hair_green_purple = gr.Slider(
            #     label="hair green<->purple ",
            #     minimum=-.8,
            #     maximum=.8,
            #     value=0,
            #     step=0.1,
            # )
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
            # requantize = gr.Checkbox(
            #     label="Requantize Latents (necessary using text prompts)",
            #     value=True,
            # )
            asian_weight = gr.Slider(
                minimum=-2.,
                value=0,
                label="Asian",
                maximum=2.,
                step=0.07,
            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown(value="""## Image Upload
                                    For best results, crop the photos like in the example pictures""", show_label=False)
                    with gr.Row():
                        base_img = gr.Image(label="Base Image", type="filepath")
                        blend_img = gr.Image(label="Image for face blending (optional)", type="filepath")
                    # gr.Markdown("## Image Examples")
                with gr.Accordion(label="Add Mask", open=False):
                    mask = gr.Image(tool="sketch", interactive=True)
                    gr.Markdown(value="Note: You must clear the mask using the rewind button every time you want to change the mask (this is a gradio bug)")
                    set_mask = gr.Button(value="Set mask")
                    gr.Text(value="this image shows the mask passed to the model when you press set mask (debugging purposes)")
                    testim = gr.Image()
                    clear_mask = gr.Button(value="Clear mask")
                    clear_mask.click(get_cleared_mask, outputs=mask)
            with gr.Row():
                gr.Examples(
                    examples=glob.glob("test_pics/*"),
                    inputs=base_img,
                    outputs=blend_img,
                    fn=set_img_from_example,
                    # cache_examples=True,
                    ) 
        with gr.Column(scale=1):
            out = gr.Image()
            rewind = gr.Slider(value=100,
                                label="Rewind back through a prompt transform: Use this to scroll through the iterations of your prompt transformation.",
                                minimum=0,
                                maximum=100)

            apply_prompts = gr.Button(value="Apply Prompts", elem_id="apply")
            clear = gr.Button(value="Clear all transformations (irreversible)", elem_id="warning")
            with gr.Accordion(label="Save Animation", open=False):
                gr.Text(value="Creates an animation of all the steps in the editing process", show_label=False)
                duration = gr.Number(value=10, label="Duration of the animation in seconds")
                extend_frames = gr.Checkbox(value=True, label="Make first and last frame longer")
                gif = gr.File(interactive=False)
                create_animation = gr.Button(value="Create Animation")
                create_animation.click(create_gif, inputs=[duration, extend_frames], outputs=gif)

        with gr.Column(scale=1):
            gr.Markdown(value="""## Text Prompting
            See readme for a prompting guide. Use the '|' symbol to separate prompts. Use the "Add mask" section to make local edits. Negative prompts are highly recommended""", show_label=False)
            positive_prompts = gr.Textbox(label="Positive prompts",
                                            value="a picture of a woman with a very big nose | a picture of a woman with a large wide nose | a woman with an extremely prominent nose")
            negative_prompts = gr.Textbox(label="Negative prompts",
                                            value="a picture of a person with a tiny nose | a picture of a person with a very thin nose")
            gen_prompts = gr.Button(value="🎲 Random prompts")
            gen_prompts.click(get_random_prompts, outputs=[positive_prompts, negative_prompts])
            with gr.Row():
                with gr.Column():
                    gr.Text(value="Prompt Editing Configuration", show_label=False)
                    with gr.Row():
                        gr.Markdown(value="## Preset Configs", show_label=False)
                    with gr.Row():
                        with gr.Column():
                            small_local = gr.Button(value="Small Masked Changes (e.g. add lipstick)", elem_id="small_local").style(full_width=False)
                        with gr.Column():
                            major_local = gr.Button(value="Major Masked Changes (e.g. change hair color or nose size)").style(full_width=False)
                        with gr.Column():
                            major_global = gr.Button(value="Major Global Changes (e.g. change race / gender").style(full_width=False)
                    iterations = gr.Slider(minimum=10,
                                            maximum=60,
                                            step=1,
                                            value=20,
                                            label="Iterations: How many steps the model will take to modify the image. Try starting small and seeing how the results turn out, you can always resume with afterwards",)
                    learning_rate = gr.Slider(minimum=4e-3,
                                            maximum=7e-1,
                                            value=1e-1,
                                            label="Learning Rate: How strong the change in each step will be (you should raise this for bigger changes (for example, changing hair color), and lower it for more minor changes. Raise if changes aren't strong enough")
                with gr.Accordion(label="Advanced Prompt Editing Options", open=False):
                    lpips_weight = gr.Slider(minimum=0,
                                            maximum=50,
                                            value=1,
                                            label="Perceptual similarity weight (Keeps areas outside of the mask looking similar to the original. Increase if the rest of the image is changing too much while you're trying to change make a localized edit")
                    reconstruction_steps = gr.Slider(minimum=0,
                                            maximum=50,
                                            value=15,
                                            step=1,
                                            label="Steps to run at the end of the optimization, optimizing only the masked perceptual loss. If the edit is changing the identity too much, this setting will run steps at the end that will 'pull' the image back towards the original identity")
                    # discriminator_steps = gr.Slider(minimum=0,
                    #                         maximum=50,
                    #                         step=1,
                    #                         value=0,
                    #                         label="Steps to run at the end, optimizing only the discriminator loss. This helps to reduce artefacts, but because the model is trained on CelebA, this will make your generations look more like generic white celebrities")
    clear.click(state.clear_transforms, outputs=[out, mask])
    asian_weight.change(state.apply_gender_vector, inputs=[asian_weight], outputs=[out, mask])
    lip_size.change(state.apply_lip_vector, inputs=[lip_size], outputs=[out, mask])
    # hair_green_purple.change(state.apply_gp_vector, inputs=[hair_green_purple], outputs=[out, mask])
    blue_eyes.change(state.apply_rb_vector, inputs=[blue_eyes], outputs=[out, mask])
    blend_weight.change(state.blend, inputs=[blend_weight], outputs=[out, mask])
    # requantize.change(state.update_requant, inputs=[requantize], outputs=[out, mask])
    base_img.change(state.update_images, inputs=[base_img, blend_img, blend_weight], outputs=[out, mask])
    blend_img.change(state.update_images, inputs=[base_img, blend_img, blend_weight], outputs=[out, mask])
    small_local.click(set_small_local, outputs=[iterations, learning_rate, lpips_weight, reconstruction_steps])
    major_local.click(set_major_local, outputs=[iterations, learning_rate, lpips_weight, reconstruction_steps])
    major_global.click(set_major_global, outputs=[iterations, learning_rate, lpips_weight, reconstruction_steps])
    apply_prompts.click(state.apply_prompts, inputs=[positive_prompts, negative_prompts, learning_rate, iterations, lpips_weight, reconstruction_steps], outputs=[out, mask])
    rewind.change(state.rewind, inputs=[rewind], outputs=[out, mask])
    set_mask.click(state.set_mask, inputs=mask, outputs=testim)
demo.queue()
demo.launch(debug=True, enable_queue=True)
