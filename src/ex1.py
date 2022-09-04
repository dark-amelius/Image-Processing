import pathlib
from PIL import Image
import numpy as np
import math
from tqdm import tqdm
import cv2

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
            # Get new coordinates

            nx = x * ws
            ny = y * hs

            # Get neighbours
            y_prev = int(math.floor(ny))
            y_next = min(y_prev+1, h-1)
            x_prev = int(math.floor(nx))
            x_next = min(x_prev+1, w-1)

            # Get Distances
            dy_next = y_next - ny
            dy_prev = 1 - dy_next
            dx_next = x_next - nx
            dx_prev = 1 - dx_next

            # Interpolate values
            p1 = image[y_next][x_prev] * dx_next + \
                image[y_next][x_next] * dx_prev
            p2 = image[y_prev][x_prev] * dx_next + \
                image[y_prev][x_next] * dx_prev
            p3 = dy_prev * p1 + dy_next * p2

            n_image[y][x] = p3
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

    # 2D Cartoon-ish image
    print("Bilinear KIKI")
    n_kiki = cv2.resize(KIKI, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
    n_kiki = Image.fromarray(n_kiki.astype(np.uint8)).convert('RGB')
    n_kiki.save(results.joinpath('kiki-binlinear.jpg'))
    print()

    # High-Res photography
    print("Bicubic FUJI")
    n_fuji = cv2.resize(FUJI, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    n_fuji = Image.fromarray(n_fuji.astype(np.uint8)).convert('RGB')
    n_fuji.save(results.joinpath('fuji-binlinear.jpg'))
    print()

    # Color blocks
    print("Bicubic COLORS")
    n_colors = cv2.resize(COLORS, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    n_colors = Image.fromarray(n_colors.astype(np.uint8)).convert('RGB')
    n_colors.save(results.joinpath('colors-bilinear.png'))
    print()


if __name__ == "__main__":
    ex1()
