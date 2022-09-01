import math
import pathlib
from PIL import Image
import numpy as np

script_location = pathlib.Path(__file__)
assets = script_location.parent.joinpath("../assets/")


FUJI = np.array(Image.open(assets.joinpath("fuji.jpg")))
KIKI = np.array(Image.open(assets.joinpath("kiki.jpg")))


def nearest_neighbor(image, width, height):
    x, y, z = image.shape
    n_image = np.zeros([width, height, z])

    sx = x/width
    sy = y/width

    for y in range(len(n_image)):
        for x in range(len(n_image[y])):
            n_x = math.floor(x * sx)
            n_y = math.floor(y * sy)
            n_image[y][x] = image[n_y][n_x]
    
    return n_image



def ex1():
    ## 2D Cartoon-ish image
    n_kiki = nearest_neighbor(KIKI, 2000, 2000)
    n_kiki = Image.fromarray(n_kiki.astype(np.uint8)).convert('RGB')
    n_kiki.save(assets.joinpath('kiki-0-order.jpg'))

    ## High-Res photography
    n_fuji = nearest_neighbor(FUJI, 4000, 4000)
    n_fuji = Image.fromarray(n_fuji.astype(np.uint8)).convert('RGB')
    n_fuji.save(assets.joinpath('fuji-0-order.jpg'))


if __name__ == "__main__":
    ex1()