import torch
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from gradient_flow_ops import replace_grad, clamp_with_grad
from torchvision import transforms
from CLIP import clip as clip_module
from torchvision import transforms
from img_processing import custom_to_pil
from vqgan_latent_ops import vector_quantize
from latent_utils import get_latent_from_path
from loaders import load_default

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, num_cutouts=32, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.num_cutouts = num_cutouts

        self.cut_pow = cut_pow # not used with pooling
        
        # Pick your own augments & their order
        augment_list = []
        # for item in args.augments[0]:
        #     if item == 'Ji':
        #         augment_list.append(K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7))
        #     elif item == 'Sh':
        #         augment_list.append(K.RandomSharpness(sharpness=0.3, p=0.5))
        #     elif item == 'Gn':
        #         augment_list.append(K.RandomGaussianNoise(mean=0.0, std=1., p=0.5))
        #     elif item == 'Pe':
        #         augment_list.append(K.RandomPerspective(distortion_scale=0.7, p=0.7))
        #     elif item == 'Ro':
        #         augment_list.append(K.RandomRotation(degrees=15, p=0.7))
        #     elif item == 'Af':
        #         augment_list.append(K.RandomAffine(degrees=15, translate=0.1, shear=5, p=0.7, padding_mode='zeros', keepdim=True)) # border, reflection, zeros
        #     elif item == 'Et':
        #         augment_list.append(K.RandomElasticTransform(p=0.7))
        #     elif item == 'Ts':
        #         augment_list.append(K.RandomThinPlateSpline(scale=0.8, same_on_batch=True, p=0.7))
        #     elif item == 'Cr':
        #         augment_list.append(K.RandomCrop(size=(self.cut_size,self.cut_size), pad_if_needed=True, padding_mode='reflect', p=0.5))
        #     elif item == 'Er':
        #         augment_list.append(K.RandomErasing(scale=(.1, .4), ratio=(.3, 1/.3), same_on_batch=True, p=0.7))
        #     elif item == 'Re':
        #         augment_list.append(K.RandomResizedCrop(size=(self.cut_size,self.cut_size), scale=(0.1,1),  ratio=(0.75,1.333), cropping_mode='resample', p=0.5))
                
        # self.augs = nn.Sequential(*augment_list)
        self.augs = nn.Identity()
        # self.noise_fac = 0.1
        # self.noise_fac = False

        # Uncomment if you like seeing the list ;)
        # print(augment_list)
        
        # Pooling
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        cutouts = []
        
        for _ in range(self.num_cutouts):            
            # Use Pooling
            cutout = (self.av_pool(input) + self.max_pool(input))/2
            cutouts.append(cutout)
            
        batch = self.augs(torch.cat(cutouts, dim=0))
        
        # if self.noise_fac:
        #     facs = batch.new_empty([self.num_cutouts, 1, 1, 1]).uniform_(0, self.noise_fac)
        #     batch = batch + facs * torch.randn_like(batch)
        return batch

class Prompt(nn.Module):
    def __init__(self, text_embedding, weight=1.):
        super().__init__()
        self.register_buffer('text_embedding', text_embedding)
    def forward(self, clip_img_embed):
        input_normed = F.normalize(clip_img_embed.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.text_embedding.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, dists).mean()

class EditableImage():
    def __init__(self, vqgan, clip, path=None, tensor_img=None) -> None:
        assert path is not None or tensor_img is not None
        self.quantize_embedding_weight = vqgan.quantize.embedding.weight
        if path:
            self.gan_latent = get_latent_from_path(path)    
        else:
            raise NotImplementedError
        # self.clip_embedding = clip.encode_image(normalize(make_cutouts(out))).float()
    # def apply_vector(self, vector, quantize=True):
    #     """Applies a vector to the VQGAN latent embedding of the image and returns the decoded image"""
    #     transformed_latent = self.gan_latent + vector
    #     # return self.perceptor
    #     z_q = vector_quantize(z.movedim(1, 3), self.quantize_embedding_weight).movedim(3, 1)
    #     return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)

class CLIPWrapper():
    def __init__(self, device, model_name="ViT-B/32") -> None:
        self.device = device
        self.tokenize = clip_module.tokenize
        self.model = (clip_module.load(model_name, jit=False)[0]
            .eval()
            .requires_grad_(False)
            .to(self.device))
        self.visual_input_res = self.model.visual.input_resolution
        self.visual_output_res = self.model.visual.output_dim
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                  std=[0.26862954, 0.26130258, 0.27577711])
        print(f"self.input_res = {self.visual_input_res}")
        self.make_cutouts = MakeCutouts(self.visual_input_res)
    def get_text_embedding(self, text):
        tokenized_text = self.tokenize(text).to(self.device)
        embedding = self.model.encode_text(tokenized_text).float()
        return embedding
    def get_normed_cutouts(self, image):
        cutouts = self.make_cutouts(image)
        norm_cutouts = self.normalize(cutouts)
        return norm_cutouts
    def get_img_embedding(self, image):
        cutouts = self.get_normed_cutouts(image)
        emb = self.model.encode_image(cutouts).float()
        return emb


class VQGAN_CLIP(nn.Module):
    def __init__(self, vqgan, clip: CLIPWrapper) -> None:
        super().__init__()
        self.clip = clip
        self.vqgan = vqgan
    def process_and_decode(self, latent):
        quant_latent = vector_quantize(latent.movedim(1, 3), self.vqgan.quantize.embedding.weight).movedim(3, 1)
        return clamp_with_grad(self.vqgan.decode(quant_latent).add(1).div(2), 0, 1)
    def get_im_text_similarity(self, generated_img, text_emb):
        # generated_img = self.vqgan.decode(gan_latent)
        clip_img_embed = self.clip.get_img_embedding(generated_img)
        norm_clip_img_embed = F.normalize(clip_img_embed.unsqueeze(1), dim=2)
        norm_text_emb = F.normalize(text_emb.unsqueeze(0), dim=2)
        text_img_similarity = norm_clip_img_embed.sub(norm_text_emb).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        # text_img_similarity = replace_grad(text_img_similarity, text_img_similarity).mean()
        return text_img_similarity.mean() # return loss

def train(model, latent, text_prompt, steps, display_every=10):
    prompt_embed = model.clip.get_text_embedding(text_prompt)
    latent = latent.clone().detach().requires_grad_()
    opt = torch.optim.Adam([latent])
    for i in tqdm(range(steps)):
        opt.zero_grad(set_to_none=True)
        generated_im = model.process_and_decode(latent)
        loss = model.get_im_text_similarity(generated_im, prompt_embed)
        if i % display_every == 0:
            plt.imshow(custom_to_pil(generated_im[0]))
            plt.show()
        loss.backward()
        opt.step()
    #with torch.no_grad():
    # with torch.inference_mode():
        # z.copy_(z.maximum(z_min).minimum(z_max))

def main(device):
    # device = torch.cuda.is_available() else "cpu"
    image_path = "./test_data/face.jpeg"
    vqgan = load_default(device)
    clip = CLIPWrapper(device)
    model = VQGAN_CLIP(vqgan, clip)
    latent = get_latent_from_path(image_path, vqgan, device=device)
    prompt = "a picture of a man"
    from transformers import CLIPProcessor, CLIPModel
    hf_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    hf_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    emb1 = clip.get_text_embedding(prompt)
    emb2 = hf_processor(text=prompt, return_tensors="pt")
    print()
    # train(model, latent, prompt, 100)
if __name__ == "__main__":
    device = "mps"
    main(device)

