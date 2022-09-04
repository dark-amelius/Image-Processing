import pathlib
from PIL import Image
import numpy as np
import math
from tqdm import tqdm
import cv2

script_location = pathlib.Path(__file__)
assets = script_location.parent.joinpath("../assets/")
results = script_location.parent.joinpath("../results/")

CAT = Image.open(assets.joinpath("cat.jpg"))
CAMERA = Image.open(assets.joinpath("cameraman.tif"))
RICE = Image.open(assets.joinpath("rice.jpg"))
# DOG = np.array(Image.open(assets.joinpath("dog.jpg")))
# DOG = CAT.convert('L')
# DOG = np.array(DOG)
# CHIHIRO = np.array(Image.open(assets.joinpath("chihiro.jpg")))
KIKI = Image.open(assets.joinpath("kiki.jpg"))
FUJI = Image.open(assets.joinpath("fuji.jpg"))


def contrast_stretching(img, is_colored=False):
    if is_colored:
        i = img.convert('HSV')
    else:
        i = img.convert('L')
    width, height = i.size
    higher = 0
    lower = 256
    if is_colored:
        for w in range(width):
            for h in range(height):
                if (i.getpixel((w, h)))[2] > higher:
                    higher = i.getpixel((w, h))[2]

        for w in range(width):
            for h in range(height):
                if (i.getpixel((w, h)))[2] < lower:
                    lower = i.getpixel((w, h))[2]

        for w in tqdm(range(width)):
            for h in range(height):
                p = i.getpixel((w, h))
                v = (p[0], p[1], int((255*(p[2] - lower))/(higher-lower)))
                i.putpixel((w, h), v)
    else:
        for w in range(width):
            for h in range(height):
                if (i.getpixel((w, h))) > higher:
                    higher = i.getpixel((w, h))

        for w in range(width):
            for h in range(height):
                if (i.getpixel((w, h))) < lower:
                    lower = i.getpixel((w, h))

        for w in tqdm(range(width)):
            for h in range(height):
                p = i.getpixel((w, h))
                p = int((255*(p - lower))/(higher-lower))
                i.putpixel((w, h), p)
    return i


def get_histogram(img, is_colored=False):
    if is_colored:
        img = img.convert('HSV')
    else:
        img = img.convert('L')
    width, height = img.size
    histogram = [0]*256
    num = [0]*256

    for i in range(255):
        num[i] = i

    for x in range(width):
        for y in range(height):
            if is_colored:
                p = img.getpixel((x, y))[2]
            else:
                p = img.getpixel((x, y))
            histogram[p] += 1

    return histogram


def cumulative_histogram(img, is_colored=False):
    histogram = get_histogram(img, is_colored)
    cumulative_hist = [0.0]*256
    width, height = img.size
    num_pixels = width*height
    alpha = int(255.0/num_pixels)
    cumulative_hist[0] = alpha*(histogram[0])

    for j in range(255):
        cumulative_hist[j] = cumulative_hist[j-1] + alpha*histogram[j]

    return cumulative_hist


def histogram_equalization(img, is_colored=False):
    if is_colored:
        img = img.convert("HSV")
    else:
        img = img.convert("L")
    histogram = get_histogram(img, is_colored)
    i = img
    cumulative_hist = [0]*256
    width, height = img.size
    num_pixels = width*height
    alpha = 255.0/num_pixels
    cumulative_hist[0] = alpha*histogram[0]

    for j in range(255):
        cumulative_hist[j] = cumulative_hist[j-1] + alpha*histogram[j]

    for w in tqdm(range(width)):
        for h in range(height):
            c = img.getpixel((w, h))
            if is_colored:
                v = (c[0], c[1], (int(cumulative_hist[c[2]])))
            else:
                i.putpixel((w, h), (int(cumulative_hist[c])))

    return i


img = histogram_equalization(RICE)
img.save(results.joinpath('rice-histo.jpg'))
print()

img = contrast_stretching(RICE)
img.save(results.joinpath('rice-stretching.jpg'))
print()

img = histogram_equalization(KIKI, is_colored=True).convert('RGB')
img.save(results.joinpath('Kiki-histo.jpg'))
print()

img = contrast_stretching(KIKI, is_colored=True).convert('RGB')
img.save(results.joinpath('Kiki-stretch.jpg'))
print()
