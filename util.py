import numpy as np
import imageio
import os


def exp_parameter():
    h = 6.626*1e-34/(2*np.pi)
    m = 1.44*1e-25
    w = 219*2*np.pi/2**0.5

    d = {
        "h": h,
        "m": m,
        "w": w,
        "a": 5.1*1e-9,
        "a0": np.sqrt(h / (m*w)),
        "N0": 1e5,
    }

    return d


def img_to_mp4(path, filelist, fps):

    writer = imageio.get_writer(os.path.join(path, 'simulation.mp4'), fps=fps)

    for file in filelist:
        im = imageio.imread(file)
        writer.append_data(im)
    writer.close()


if __name__ == "__main__":

    import glob
    import os

    filelist = glob.glob("./img/frame_*.png")

    filelist.sort(key=lambda s: int(s.split("frame_")[1].split(".")[0]))
    img_to_mp4("./", filelist, fps=15)
