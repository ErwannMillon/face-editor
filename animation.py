import imageio
import glob
import os


def clear_img_dir(img_dir):
    for filename in glob.glob(img_dir + "/*"):
        os.remove(filename)


def create_gif(
    total_duration, extend_frames, folder="./img_history", gif_name="face_edit.gif"
):
    images = []
    paths = list(sorted(glob.glob(folder + "/*")))
    print(paths)
    frame_duration = total_duration / len(paths)
    print(len(paths), "frame dur", frame_duration)
    durations = [frame_duration] * len(paths)
    if extend_frames:
        durations[0] = 1.5
        durations[-1] = 3
    for file_name in paths:
        if file_name.endswith(".png"):
            images.append(imageio.imread(file_name))
    imageio.mimsave(gif_name, images, duration=durations)
    return gif_name


if __name__ == "__main__":
    create_gif()
