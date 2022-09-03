import pathlib
from PIL import Image
import numpy as np
import math
from tqdm import tqdm

script_location = pathlib.Path(__file__)
assets = script_location.parent.joinpath("../assets/")
results = script_location.parent.joinpath("../results/")


FUJI = np.array(Image.open(assets.joinpath("fuji.jpg")))
KIKI = np.array(Image.open(assets.joinpath("kiki.jpg")))
COLORS = np.array(Image.open(assets.joinpath("colors.png")))


def nearest_neighbor(image, scale):
    h, w, c = image.shape
    width = w * scale
    height = h * scale
    ws = w/width
    hs = h/height

    n_image = np.zeros([height, width, c])

    for y in tqdm(range(height)):
        for x in range(width):
            n_x = math.floor(x * ws)
            n_y = math.floor(y * hs)
            n_image[y][x] = image[n_y][n_x]
    return n_image


def bilinear_interpol(image, scale):
    h, w, c = image.shape

    width = w * scale
    height = h * scale
    ws = w/width
    hs = h/height

    n_image = np.zeros([height, width, c])

    for y in tqdm(range(height)):
        for x in range(width):
            n_y = y * hs
            n_x = x * ws

            prev_y = math.floor(n_y)
            prev_x = math.floor(n_x)

            next_y = min(h-1, math.ceil(n_y))
            next_x = min(w-1, math.ceil(n_x))

            if (next_x == prev_x) and (next_y == prev_y):
                np_f = image[int(n_y), int(n_x)]
            elif (next_x == prev_x):
                p1 = image[int(n_y), int(prev_x)]
                p2 = image[int(n_y), int(next_x)]
                np_f = (p1 * (next_y - n_y)) + (p2 * (n_y - prev_y))
            elif (next_y == prev_y):
                p1 = image[int(prev_y), int(n_x)]
                p2 = image[int(next_y), int(n_x)]
                np_f = (p1 * (next_x - n_x)) + (p2 * (n_x - prev_x))
            else:
                # Neighbouring pixels:
                p1 = image[prev_y, prev_x]
                p2 = image[next_y, prev_x]
                p3 = image[prev_y, next_x]
                p4 = image[next_y, next_x]

                # Interpolate in the x-axis
                np_1 = p1 * (next_x - n_x) + p2 * (n_x - prev_x)
                np_2 = p3 * (next_x - n_x) + p4 * (n_x - prev_x)

                # Interpolate in the y-axis
                np_f = np_1 * (next_y - n_y) + np_2 * (n_y - prev_y)
            n_image[y][x] = np_f
    return n_image


def ex1():

    # 2D Cartoon-ish image
    print("Nearest neighbor KIKI")
    n_kiki = nearest_neighbor(KIKI, 6)
    n_kiki = Image.fromarray(n_kiki.astype(np.uint8)).convert('RGB')
    n_kiki.save(results.joinpath('kiki-0-order.jpg'))
    print()

    # High-Res photography
    print("Nearest neighbor FUJI")
    n_fuji = nearest_neighbor(FUJI, 4)
    n_fuji = Image.fromarray(n_fuji.astype(np.uint8)).convert('RGB')
    n_fuji.save(results.joinpath('fuji-0-order.jpg'))
    print()

    # Color blocks
    print("Nearest neighbor COLORS")
    n_colors = nearest_neighbor(COLORS, 4)
    n_colors = Image.fromarray(n_colors.astype(np.uint8)).convert('RGB')
    n_colors.save(results.joinpath('colors-0-order.png'))
    print()

    # 2D Cartoon-ish image
    print("Bilinear KIKI")
    n_kiki = bilinear_interpol(KIKI, 6)
    n_kiki = Image.fromarray(n_kiki.astype(np.uint8)).convert('RGB')
    n_kiki.save(results.joinpath('kiki-binlinear.jpg'))
    print()

    # High-Res photography
    print("Bilinear FUJI")
    n_fuji = bilinear_interpol(FUJI, 4)
    n_fuji = Image.fromarray(n_fuji.astype(np.uint8)).convert('RGB')
    n_fuji.save(results.joinpath('fuji-binlinear.jpg'))
    print()

    # Color blocks
    print("Bilinear COLORS")
    n_colors = bilinear_interpol(COLORS, 4)
    n_colors = Image.fromarray(n_colors.astype(np.uint8)).convert('RGB')
    n_colors.save(results.joinpath('colors-bilinear.png'))
    print()


if __name__ == "__main__":
    ex1()
