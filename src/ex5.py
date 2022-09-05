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
LENA = np.array(Image.open(assets.joinpath("Lenna.jpg")))

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


def rotate(image, angle):
    h, w, c = image.shape

    width = w
    height = h

    a0 = math.cos(angle)
    a1 = math.sin(angle)
    b0 = -a1
    b1 = a0

    n_image = np.zeros([height, width, c])
    for y in tqdm(range(height)):
        for x in range(width):

            nx = x
            ny = y

            nx2 = math.floor(a0 * (nx - w/2) + a1 * (ny - h/2) + w/2)
            ny2 = math.floor(b0 * (nx - w/2) + b1 * (ny - h/2) + h/2)

            nx = int(nx2)
            ny = int(ny2)

            if (nx > 0 and nx < w) and (ny > 0 and ny < h):
                n_image[y][x] = image[ny][nx]
            else:
                n_image[y][x] = 0
    return n_image


def ex5():
    print("Double KIKI")
    n_kiki = nearest_neighbor(KIKI, 2)
    n_kiki = Image.fromarray(n_kiki.astype(np.uint8)).convert('RGB')
    n_kiki.save(results.joinpath('kiki-double.jpg'))
    print()

    print("Triple KIKI")
    n_kiki = nearest_neighbor(KIKI, 3)
    n_kiki = Image.fromarray(n_kiki.astype(np.uint8)).convert('RGB')
    n_kiki.save(results.joinpath('kiki-triple.jpg'))
    print()

    print("Double LENA")
    n_lena = nearest_neighbor(LENA, 2)
    n_lena = Image.fromarray(n_lena.astype(np.uint8)).convert('RGB')
    n_lena.save(results.joinpath('lena-double.jpg'))
    print()

    print("Triple LENA")
    n_lena = nearest_neighbor(LENA, 3)
    n_lena = Image.fromarray(n_lena.astype(np.uint8)).convert('RGB')
    n_lena.save(results.joinpath('lena-triple.jpg'))
    print()

    print("45 Lena Nearest NEighbor")
    n_lena = rotate(LENA, math.pi/4)
    n_lena = nearest_neighbor(n_lena, 2)
    n_lena = Image.fromarray(n_lena.astype(np.uint8)).convert('RGB')
    n_lena.save(results.joinpath('lena-45-nearest.jpg'))
    print()

    print("5 Lena Nearest Neighbor")
    n_lena = rotate(LENA, math.pi/36)
    n_lena = nearest_neighbor(n_lena, 2)
    n_lena = Image.fromarray(n_lena.astype(np.uint8)).convert('RGB')
    n_lena.save(results.joinpath('lena-5-nearest.jpg'))
    print()

    print("45 Lena Nearest NEighbor")
    n_lena = rotate(LENA, math.pi/4)
    n_lena = nearest_neighbor(n_lena, 3)
    n_lena = Image.fromarray(n_lena.astype(np.uint8)).convert('RGB')
    n_lena.save(results.joinpath('lena-45-nearest3.jpg'))
    print()

    print("5 Lena Nearest Neighbor")
    n_lena = rotate(LENA, math.pi/36)
    n_lena = nearest_neighbor(n_lena, 3)
    n_lena = Image.fromarray(n_lena.astype(np.uint8)).convert('RGB')
    n_lena.save(results.joinpath('lena-5-nearest3.jpg'))
    print()


    print("Double KIKI")
    n_kiki = bilinear_interpol(KIKI, 2)
    n_kiki = Image.fromarray(n_kiki.astype(np.uint8)).convert('RGB')
    n_kiki.save(results.joinpath('kiki-double-bi.jpg'))
    print()

    print("Triple KIKI")
    n_kiki = bilinear_interpol(KIKI, 3)
    n_kiki = Image.fromarray(n_kiki.astype(np.uint8)).convert('RGB')
    n_kiki.save(results.joinpath('kiki-triple-bi.jpg'))
    print()

    print("Double LENA")
    n_lena = bilinear_interpol(LENA, 2)
    n_lena = Image.fromarray(n_lena.astype(np.uint8)).convert('RGB')
    n_lena.save(results.joinpath('lena-double-bi.jpg'))
    print()

    print("Triple LENA")
    n_lena = bilinear_interpol(LENA, 3)
    n_lena = Image.fromarray(n_lena.astype(np.uint8)).convert('RGB')
    n_lena.save(results.joinpath('lena-triple-bi.jpg'))
    print()

    print("45 Lena Bilinear")
    n_lena = rotate(LENA, math.pi/4)
    n_lena = bilinear_interpol(n_lena, 2)
    n_lena = Image.fromarray(n_lena.astype(np.uint8)).convert('RGB')
    n_lena.save(results.joinpath('lena-45-bilinear-bi.jpg'))
    print()

    print("5 Lena Bilinear")
    n_lena = rotate(LENA, math.pi/36)
    n_lena = bilinear_interpol(n_lena, 2)
    n_lena = Image.fromarray(n_lena.astype(np.uint8)).convert('RGB')
    n_lena.save(results.joinpath('lena-5-bilinear-bi.jpg'))
    print()

    print("45 Lena Bilinear")
    n_lena = rotate(LENA, math.pi/4)
    n_lena = bilinear_interpol(n_lena, 3)
    n_lena = Image.fromarray(n_lena.astype(np.uint8)).convert('RGB')
    n_lena.save(results.joinpath('lena-45-bilinear3-bi.jpg'))
    print()

    print("5 Lena Bilinear")
    n_lena = rotate(LENA, math.pi/36)
    n_lena = bilinear_interpol(n_lena, 3)
    n_lena = Image.fromarray(n_lena.astype(np.uint8)).convert('RGB')
    n_lena.save(results.joinpath('lena-5-bilinear3-bi.jpg'))
    print()


if __name__ == "__main__":
    ex5()
