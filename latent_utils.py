from img_processing import custom_to_pil, preprocess, preprocess_vqgan
import matplotlib.pyplot as plt
import PIL
def get_latent_from_path(path, model, device="mps"):
    x = preprocess(PIL.Image.open(path), target_image_size=256, map_dalle=False)
    
    x = x.to(device)

    print(device)
    x_processed = preprocess_vqgan(x).to(device)
    x_latent, _, [_, _, indices] = model.encode(x_processed)
    print(x_latent)
    return x_latent
    
def show_latent(model, latent, device="mps"):
    dec = model.decode(latent.to(device))
    im = custom_to_pil(dec[0])
    plt.figure(figsize=(3, 3))
    plt.imshow(im)
    return im