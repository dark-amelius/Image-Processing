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


def cv_contrast_stretch(image):
    g = (image - np.amin(image) / np.amax(image) - np.amin(image))
    return Image.fromarray(g.astype('uint8'))

def contrast_stretching(img):

    i = img.convert('L')
    width, height = i.size
    higher = 0
    lower = 256

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


def get_histogram(img):

    img = img.convert('L')
    width, height = img.size
    histogram = [0]*256
    num = [0]*256

    for i in range(255):
        num[i] = i

    for x in range(width):
        for y in range(height):
            p = img.getpixel((x, y))
            histogram[p] += 1

    return histogram


def cumulative_histogram(img):
    histogram = get_histogram(img)
    cumulative_hist = [0.0]*256
    width, height = img.size
    num_pixels = width*height
    alpha = int(255.0/num_pixels)
    cumulative_hist[0] = alpha*(histogram[0])

    for j in range(255):
        cumulative_hist[j] = cumulative_hist[j-1] + alpha*histogram[j]

    return cumulative_hist


def histogram_equalization(img):
    img = img.convert("L")
    histogram = get_histogram(img)
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
            i.putpixel((w, h), (int(cumulative_hist[c])))

    return i


img = histogram_equalization(RICE)
img.save(results.joinpath('rice-histo.jpg'))
print()

img = contrast_stretching(RICE)
img.save(results.joinpath('rice-stretching.jpg'))
print()

