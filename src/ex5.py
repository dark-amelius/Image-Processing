import pathlib
from PIL import Image
import numpy as np
import math
from tqdm import tqdm
import cv2
from scipy import interpolate

script_location = pathlib.Path(__file__)
assets = script_location.parent.joinpath("../assets/")
results = script_location.parent.joinpath("../results/")


FUJI = np.array(Image.open(assets.joinpath("fuji.jpg")))
KIKI = np.array(Image.open(assets.joinpath("kiki.jpg")))
COLORS = np.array(Image.open(assets.joinpath("colors.png")))


def scale(image, a, kind):
    h, w, c = image.shape
    x, y = np.mgrid[0:h, 0:w]
    print(x)
    x1 = [[0 for i in range(w)] for j in range(w)]
    y1 = [[0 for i in range(w)] for j in range(h)]
    
    for i in range(h):
        for j in range(w):
            x1[i][j] = (a * x[i][j])
            y1[i][j] = (y[i][j]/a)
    f = interpolate.RectBivariateSpline(x, y, image)
    return f


def ex5():
    print("Double linear kiki")
    n_kiki = scale(KIKI, 2, 'linear')
    n_kiki = Image.fromarray(n_kiki.astype(np.uint8)).convert('RGB')
    n_kiki.save(results.joinpath('kiki-scale-1.jpg'))
    print()


if __name__ == "__main__":
    ex5()
