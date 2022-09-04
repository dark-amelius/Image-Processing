import pathlib
from PIL import Image
import numpy as np
import math
from tqdm import tqdm
import cv2
import random

script_location = pathlib.Path(__file__)
assets = script_location.parent.joinpath("../assets/")
results = script_location.parent.joinpath("../results/")

CAT = Image.open(assets.joinpath("cat.jpg"))
RICE = Image.open(assets.joinpath("rice.jpg"))
TEST = Image.open(assets.joinpath("testnoise.jpg"))
# DOG = np.array(Image.open(assets.joinpath("dog.jpg")))
# DOG = CAT.convert('L')
# DOG = np.array(DOG)
# CHIHIRO = np.array(Image.open(assets.joinpath("chihiro.jpg")))


def is_contaminated(percent):
    return random.randrange(100) < percent


def add_salt_and_pepper_noise(img, contamination_percentage):
    i = img.convert('L')
    width, height = i.size
    higher = 0
    lower = 256

    for w in tqdm(range(width)):
        for h in range(height):
            if is_contaminated(contamination_percentage/2):
                i.putpixel((w, h), 255)
            if is_contaminated(contamination_percentage/2):
                i.putpixel((w, h), 0)

    return i


def add_gaussian_noise(img):
    i = img
    i = np.array(i)
    w, h, c = i.shape

    noise = np.random.normal(0, 0.1**0.5, (w, h, c))
    noise.reshape(w,h,c)
    i = i + noise
    return Image.fromarray(i.astype('uint8')).convert('L')

def add_both_noises(img, sp):
    img2 = img.copy()
    img2 = add_gaussian_noise(img2)
    img2 = add_salt_and_pepper_noise(img2, sp)
    return img2

def ftAlphaTrimmedMean(image):
    # deep copy
    img = image.copy()

    # Get image height and width
    height, width = image.shape

    height = height-1
    width = width-1

    p = 0

    # loop through
    for i in tqdm(range(1, height)):
        for j in range(1, width):
            arr = []

            p = image[i-1][j-1]
            arr.append(p)

            p = image[i-1][j]
            arr.append(p)

            p = image[i-1][j+1]
            arr.append(p)

            p = image[i][j-1]
            arr.append(p)

            p = image[i][j]
            arr.append(p)

            p = image[i][j+1]
            arr.append(p)

            p = image[i+1][j-1]
            arr.append(p)

            p = image[i+1][j]
            arr.append(p)

            p = image[i+1][j+1]
            arr.append(p)

            arr = np.sort(arr)

            middle = int((len(arr)-1)/2)

            total = 0

            total += arr[middle-2]
            total += arr[middle-1]
            total += arr[middle]
            total += arr[middle+1]
            total += arr[middle+2]

            total = int(total/5)

            # set pixel value back to image
            img[i][j] = total

    return img


img = add_both_noises(CAT, 0.1)
img.save(results.joinpath('cat-noised.jpg'))

img = ftAlphaTrimmedMean(np.array(img))
img = Image.fromarray(img.astype(np.uint8))
img.save(results.joinpath('filtered.jpg'))
print()
