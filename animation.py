import imageio
import glob
import os

def clear_img_dir(img_dir):
    if not os.path.exists(img_dir):
        os.mkdir("img_history")
        os.mkdir(img_dir)
    for filename in glob.glob(img_dir+"/*"):
        os.remove(filename)
    


if __name__ == "__main__":
    # clear_img_dir()
    create_gif()
# make_animation()