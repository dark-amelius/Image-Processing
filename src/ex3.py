import pathlib
from PIL import Image, ImageFilter
import numpy as np
import math
from tqdm import tqdm
import cv2
import random
import scipy

script_location = pathlib.Path(__file__)
assets = script_location.parent.joinpath("../assets/")
results = script_location.parent.joinpath("../results/")

CAT = Image.open(assets.joinpath("cat.jpg"))
RICE = Image.open(assets.joinpath("rice.jpg"))
TEST = Image.open(assets.joinpath("testnoise.jpg"))
FUJI = Image.open(assets.joinpath("fuji.jpg"))
LYNCH = Image.open(assets.joinpath("david-lynch.jpg"))
# DOG = np.array(Image.open(assets.joinpath("dog.jpg")))
# DOG = CAT.convert('L')
# DOG = np.array(DOG)
# CHIHIRO = np.array(Image.open(assets.joinpath("chihiro.jpg")))


def is_contaminated(percent):
    return random.randrange(100) < percent


def add_salt_and_pepper_noise(img, contamination_percentage):
    i = img.copy()
    width, height = i.size
    higher = 0
    lower = 256

    for w in tqdm(range(width)):
        for h in range(height):
            if is_contaminated(contamination_percentage/2):
                i.putpixel((w, h), (255, 255, 255))
            if is_contaminated(contamination_percentage/2):
                i.putpixel((w, h), (0, 0, 0))

    return i


def add_gaussian_noise(img):
    i = img
    i = np.array(i)
    w, h, c = i.shape

    noise = np.random.normal(0, 0.1**0.5, (w, h, c))
    noise.reshape(w, h, c)
    i = i + noise
    return Image.fromarray(i.astype('uint8'))


def add_both_noises(img, sp):
    img2 = img.copy()
    img2 = add_gaussian_noise(img2)
    img2 = add_salt_and_pepper_noise(img2, sp)
    return img2


def ftAlphaTrimmedMean(image):
    # deep copy
    img = image.copy()
    img = Image.fromarray(img.astype('uint8'))
    img = img.convert('HSV')
    img = np.array(img)
    # Get image height and width
    height, width, c = image.shape

    height = height-1
    width = width-1

    p = 0

    # loop through
    for i in tqdm(range(1, height)):
        for j in range(1, width):
            arr = []

            p = image[i-1][j-1][2]
            arr.append(p)

            p = image[i-1][j][2]
            arr.append(p)

            p = image[i-1][j+1][2]
            arr.append(p)

            p = image[i][j-1][2]
            arr.append(p)

            p = image[i][j][2]
            arr.append(p)

            p = image[i][j+1][2]
            arr.append(p)

            p = image[i+1][j-1][2]
            arr.append(p)

            p = image[i+1][j][2]
            arr.append(p)

            p = image[i+1][j+1][2]
            arr.append(p)

            arr = np.sort(arr)

            middle = int((len(arr)-1)/2)

            total = 0

            total += arr[middle-2]
            total += arr[middle-1]
            total += arr[middle]
            total += arr[middle+1]
            total += arr[middle+2]

            total = total/5

            # set pixel value back to image
            img[i][j][2] = int(total)
    
    
    img = Image.fromarray(img.astype('uint8'), mode='HSV')
    img = img.convert('RGB')
    img = np.array(img)

    return img


def snr(reference, test):
    h, w = reference.shape
    reference = reference.copy().astype('float')
    test = test.copy().astype('float')
    sum1 = 0
    sum2 = 0
    for y in tqdm(range(h)):
        for x in range(w):
            sum1 += (reference[y][x] ** 2)
            sum2 += ((reference[y][x] - test[y][x]) ** 2)
    r = sum1 / sum2
    return 10 * np.log10(r)


def psnr(reference, test):
    h, w = reference.shape
    reference = reference.copy().astype('float')
    test = test.copy().astype('float')

    term1 = np.max(reference) ** 2
    sum1 = 0
    for y in tqdm(range(h)):
        for x in range(w):
            sum1 += ((reference[y][x] - test[y][x]) ** 2)
    sum1 = sum1/(h*w)
    return 10 * np.log10(term1/sum1)


def unsharp_2(img, radius=2, percent=150):
    im2 = Image.fromarray(img.astype('uint8'))
    im2 = im2.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent))
    im2 = np.array(im2)
    return im2


img = add_both_noises(FUJI, 0.1)
img.save(results.joinpath('fuji-noised.jpg'))

img2 = ftAlphaTrimmedMean(np.array(img))
img2 = Image.fromarray(img2.astype(np.uint8))
img2.save(results.joinpath('filtered-fuji.jpg'))

snr_v = snr(np.array(img.convert('L')), np.array(img2.convert('L')))
psnr_v = psnr(np.array(img.convert('L')), np.array(img2.convert('L')))
print(f'SNR: {snr_v}')
print(f'PSNR: {psnr_v}')

img3 = np.array(FUJI)
img3 = unsharp_2(img3)
img3 = Image.fromarray(img3.astype('uint8')).save(
    results.joinpath('unsharped-fuji.jpg'))

img = add_both_noises(LYNCH, 0.1)
img.save(results.joinpath('lynch-noised.jpg'))

img2 = ftAlphaTrimmedMean(np.array(img))
img2 = Image.fromarray(img2.astype(np.uint8))
img2.save(results.joinpath('filtered-lynch.jpg'))

snr_v = snr(np.array(img.convert('L')), np.array(img2.convert('L')))
psnr_v = psnr(np.array(img.convert('L')), np.array(img2.convert('L')))
print(f'SNR: {snr_v}')
print(f'PSNR: {psnr_v}')

img3 = np.array(LYNCH)
img3 = unsharp_2(img3)
img3 = Image.fromarray(img3.astype('uint8')).save(results.joinpath('unsharped-lynch.jpg'))
img3 = np.array(LYNCH)
img3 = unsharp_2(img3, percent=200)
img3 = Image.fromarray(img3.astype('uint8')).save(results.joinpath('unsharped-lynch2.jpg'))