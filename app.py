import glob
import os
import sys

import wandb
import torch

from configs import set_major_global, set_major_local, set_preset, set_small_local
import uuid 
# print()'
sys.path.append("taming-transformers")

import gradio as gr
from transformers import CLIPModel, CLIPProcessor
from lpips import LPIPS

import edit
from backend import ImagePromptOptimizer, ProcessorGradientFlow
from ImageState import ImageState
from loaders import load_default
# from animation import create_gif
from prompts import get_random_prompts

device = "cuda" if torch.cuda.is_available() else "cpu"

global vqgan
vqgan = load_default(device)
vqgan.eval()
processor = ProcessorGradientFlow(device=device)
# clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
lpips_fn = LPIPS(net='vgg').to(device)
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
promptoptim = ImagePromptOptimizer(vqgan, clip, processor, lpips_fn=lpips_fn, quantize=True)
def set_img_from_example(state, img):
    return state.update_images(img, img, 0)
def get_cleared_mask():
    return gr.Image.update(value=None)
    # mask.clear()

class StateWrapper:
    def create_gif(state, *args, **kwargs):
        return state, state[0].create_gif(*args, **kwargs)
    def apply_asian_vector(state, *args, **kwargs):
        return state, *state[0].apply_asian_vector(*args, **kwargs)
    def apply_gp_vector(state, *args, **kwargs):
        return state, *state[0].apply_gp_vector(*args, **kwargs)
    def apply_lip_vector(state, *args, **kwargs):
        return state, *state[0].apply_lip_vector(*args, **kwargs)
    def apply_prompts(state, *args, **kwargs):
        print(state[1])
        for image in state[0].apply_prompts(*args, **kwargs):
            yield state, *image
    def apply_rb_vector(state, *args, **kwargs):
        return state, *state[0].apply_rb_vector(*args, **kwargs)
    def blend(state, *args, **kwargs):
        return state, *state[0].blend(*args, **kwargs)
    def clear_transforms(state, *args, **kwargs):
        return state, *state[0].clear_transforms(*args, **kwargs)
    def init_transforms(state, *args, **kwargs):
        return state, *state[0].init_transforms(*args, **kwargs)
    def prompt_optim(state, *args, **kwargs):
        return state, *state[0].prompt_optim(*args, **kwargs)
    def rescale_mask(state, *args, **kwargs):
        return state, *state[0].rescale_mask(*args, **kwargs)
    def rewind(state, *args, **kwargs):
        return state, *state[0].rewind(*args, **kwargs)
    def set_mask(state, *args, **kwargs):
        return state, state[0].set_mask(*args, **kwargs)
    def update_images(state, *args, **kwargs):
        return state, *state[0].update_images(*args, **kwargs)
    def update_requant(state, *args, **kwargs):
        return state, *state[0].update_requant(*args, **kwargs)
with gr.Blocks(css="styles.css") as demo:
    id = gr.State(str(uuid.uuid4()))
    state = gr.State([ImageState(vqgan, promptoptim), str(uuid.uuid4())])
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(value="""## Image Upload
                                    For best results, crop the photos like in the example pictures""", show_label=False)
                    with gr.Row():
                        base_img = gr.Image(label="Base Image", type="filepath")
                        blend_img = gr.Image(label="Image for face blending (optional)", type="filepath")
                with gr.Accordion(label="Add Mask", open=False):
                    mask = gr.Image(tool="sketch", interactive=True)
                    gr.Markdown(value="Note: You must clear the mask using the rewind button every time you want to change the mask (this is a gradio issue)")
                    set_mask = gr.Button(value="Set mask")
                    gr.Text(value="this image shows the mask passed to the model when you press set mask (debugging purposes)")
                    testim = gr.Image()
            with gr.Row():
                gr.Examples(
                    examples=glob.glob("test_pics/*"),
                    inputs=base_img,
                    outputs=blend_img,
                    fn=set_img_from_example,
                    ) 
        with gr.Column(scale=1):
            out = gr.Image()
            rewind = gr.Slider(value=100,
                                label="Rewind back through a prompt transform: Use this to scroll through the iterations of your prompt transformation.",
                                minimum=0,
                                maximum=100)

            apply_prompts = gr.Button(variant="primary", value="🎨 Apply Prompts", elem_id="apply")
            clear = gr.Button(value="❌ Clear all transformations (irreversible)", elem_id="warning")
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
            with gr.Accordion(label="💾 Save Animation", open=False):
                gr.Text(value="Creates an animation of all the steps in the editing process", show_label=False)
                duration = gr.Number(value=10, label="Duration of the animation in seconds")
                extend_frames = gr.Checkbox(value=True, label="Make first and last frame longer")
                gif = gr.File(interactive=False)
                create_animation = gr.Button(value="Create Animation")
                create_animation.click(StateWrapper.create_gif, inputs=[state, duration, extend_frames], outputs=[state, gif])

        with gr.Column(scale=1):
            gr.Markdown(value="""## ✍️ Prompt Editing
            See readme for a prompting guide. Use the '|' symbol to separate prompts. Use the "Add mask" section to make local edits (Remember to click Set Mask!). Negative prompts are highly recommended""", show_label=False)
            positive_prompts = gr.Textbox(label="Positive prompts",
                                            value="A picture of a handsome man | a picture of a masculine man",)
            negative_prompts = gr.Textbox(label="Negative prompts",
                                            value="a picture of a woman | a picture of a feminine person")
            gen_prompts = gr.Button(value="🎲 Random prompts")
            gen_prompts.click(get_random_prompts, outputs=[positive_prompts, negative_prompts])
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        gr.Markdown(value="## ⚙ Prompt Editing Config", show_label=False)
                    with gr.Accordion(label="Config Tutorial", open=False):
                        gr.Markdown(value="""
                        - If results are not changing enough, increase the learning rate or decrease the perceptual loss weight
                        - To make local edits, use the 'Add Mask' section
                        - If using a mask and the image is changing too much outside of the masked area, try increasing the perceptual loss weight or lowering the learning rate
                        - Use the rewind slider to scroll through the iterations of your prompt transformation, you can resume editing from any point in the history. 
                        - I recommend starting prompts with 'a picture of a'
                        - To avoid shifts in gender, you can use 'a person' instead of 'a man' or 'a woman', especially in the negative prompts. 
                        - The more 'out-of-domain' the prompts are, the more you need to increase the learning rate and decrease the perceptual loss weight. For example, trying to make a black person have platinum blond hair is more out-of-domain than the same transformation on a caucasian person. 
                        - Example: Higher config values, like learning rate: 0.7, perceptual loss weight: 35 can be used to make major out-of-domain changes.
                        """)
                    with gr.Row():
                        # with gr.Column():
                        presets = gr.Dropdown(value="Select a preset", label="Preset Configs", choices=["Small Masked Changes (e.g. add lipstick)", "Major Masked Changes (e.g. change hair color or nose size)", "Major Global Changes (e.g. change race / gender"])
                    iterations = gr.Slider(minimum=10,
                                            maximum=60,
                                            step=1,
                                            value=20,
                                            label="Iterations: How many steps the model will take to modify the image. Try starting small and seeing how the results turn out, you can always resume with afterwards",)
                    learning_rate = gr.Slider(minimum=4e-3,
                                            maximum=1,
                                            value=1e-1,
                                            label="Learning Rate: How strong the change in each step will be (you should raise this for bigger changes (for example, changing hair color), and lower it for more minor changes. Raise if changes aren't strong enough")
                    lpips_weight = gr.Slider(minimum=0,
                                            maximum=50,
                                            value=1,
                                            label="Perceptual Loss weight (Keeps areas outside of the mask looking similar to the original. Increase if the rest of the image is changing too much while you're trying to change make a localized edit")
                    reconstruction_steps = gr.Slider(minimum=0,
                                            maximum=50,
                                            value=3,
                                            step=1,
                                            label="Steps to run at the end of the optimization, optimizing only the masked perceptual loss. If the edit is changing the identity too much, this setting will run steps at the end that 'pull' the image back towards the original identity")
                    # discriminator_steps = gr.Slider(minimum=0,
                    #                         maximum=50,
                    #                         step=1,
                    #                         value=0,
                    #                         label="Steps to run at the end, optimizing only the discriminator loss. This helps to reduce artefacts, but because the model is trained on CelebA, this will make your generations look more like generic white celebrities")
    clear.click(StateWrapper.clear_transforms, inputs=[state], outputs=[state, out, mask])
    asian_weight.change(StateWrapper.apply_asian_vector, inputs=[state, asian_weight], outputs=[state, out, mask])
    lip_size.change(StateWrapper.apply_lip_vector, inputs=[state, lip_size], outputs=[state, out, mask])
    # hair_green_purple.change(StateWrapper.apply_gp_vector, inputs=[state, hair_green_purple], outputs=[state, out, mask])
    blue_eyes.change(StateWrapper.apply_rb_vector, inputs=[state, blue_eyes], outputs=[state, out, mask])
    blend_weight.change(StateWrapper.blend, inputs=[state, blend_weight], outputs=[state, out, mask])
    # requantize.change(StateWrapper.update_requant, inputs=[state, requantize], outputs=[state, out, mask])
    base_img.change(StateWrapper.update_images, inputs=[state, base_img, blend_img, blend_weight], outputs=[state, out, mask])
    blend_img.change(StateWrapper.update_images, inputs=[state, base_img, blend_img, blend_weight], outputs=[state, out, mask])
    # small_local.click(set_small_local, outputs=[iterations, learning_rate, lpips_weight, reconstruction_steps])
    # major_local.click(set_major_local, outputs=[iterations, learning_rate, lpips_weight, reconstruction_steps])
    # major_global.click(set_major_global, outputs=[iterations, learning_rate, lpips_weight, reconstruction_steps])
    apply_prompts.click(StateWrapper.apply_prompts, inputs=[state, positive_prompts, negative_prompts, learning_rate, iterations, lpips_weight, reconstruction_steps], outputs=[state, out, mask])
    rewind.change(StateWrapper.rewind, inputs=[state, rewind], outputs=[state, out, mask])
    set_mask.click(StateWrapper.set_mask, inputs=[state, mask], outputs=[state, testim])
    presets.change(set_preset, inputs=[presets], outputs=[iterations, learning_rate, lpips_weight, reconstruction_steps])
demo.queue()
demo.launch(debug=True, enable_queue=True)
