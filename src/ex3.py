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


img = add_salt_and_pepper_noise(CAT, 0.1)
img.save(results.joinpath('cat-salt_and_pepper.jpg'))
print()
