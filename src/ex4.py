import pathlib
from PIL import Image
import numpy as np
import math
from tqdm import tqdm

script_location = pathlib.Path(__file__)
assets = script_location.parent.joinpath("../assets/")
results = script_location.parent.joinpath("../results/")

RDJ = np.array(Image.open(assets.joinpath("rdj.jpg")).convert('L'))
KYUBEY = np.array(Image.open(assets.joinpath("kyubey.png")).convert('L'))

def random_dithering(image,bitsperpixel):
	h,w = image.shape
	dithered = np.zeros([h, w])
	
	levels = 2**bitsperpixel
	inc = 255 / (levels-1)
	
	for y in tqdm(range(h)):
		for x in range(w):
			rand_values = np.sort(np.random.randint(0,255,levels-1))
			for i in range(levels-1):
				if (image[y][x] < rand_values[i]):
					dithered[y][x] = i * inc
					break
				elif (i == (levels-2)):
					dithered[y][x] = 255
					break
	
	return dithered

def int8_truncate(n):
	if n > 255:
		return 255
	elif n < 0:
		return 0
	else:
		return n

def floyd_steinberg_dithering(image,bitsperpixel):
	h,w = image.shape
	dithered = np.copy(image)
	
	levels = 2**bitsperpixel

	for y in tqdm(range(0,h-1)):
		for x in range(1,w-1):
			oldpixel = dithered[y][x]
			newpixel = int8_truncate(np.round((levels-1) * oldpixel / 255) * (255 / (levels-1)))
			
			dithered[y][x] = newpixel
			error = oldpixel - newpixel
			
			dithered[y  ][x+1] = int8_truncate(dithered[y  ][x+1] + error * 7/16.0)
			dithered[y+1][x-1] = int8_truncate(dithered[y+1][x-1] + error * 3/16.0)
			dithered[y+1][x  ] = int8_truncate(dithered[y+1][x  ] + error * 5/16.0)
			dithered[y+1][x+1] = int8_truncate(dithered[y+1][x+1] + error * 1/16.0)
	
	return dithered

#random dithering to 2 bits/pixel
res = random_dithering(RDJ,2)
res = Image.fromarray(res.astype(np.uint8)).convert('RGB')
res.save(results.joinpath('rdj_2bpp_random.jpg'))
print()

res = random_dithering(KYUBEY,2)
res = Image.fromarray(res.astype(np.uint8)).convert('RGB')
res.save(results.joinpath('kyubey_2bpp_random.jpg'))
print()

#random dithering to 3 bits/pixel
res = random_dithering(RDJ,3)
res = Image.fromarray(res.astype(np.uint8)).convert('RGB')
res.save(results.joinpath('rdj_3bpp_random.jpg'))
print()

res = random_dithering(KYUBEY,3)
res = Image.fromarray(res.astype(np.uint8)).convert('RGB')
res.save(results.joinpath('kyubey_3bpp_random.jpg'))
print()

#FS dithering to 2 bits/pixel
res = floyd_steinberg_dithering(RDJ,2)
res = Image.fromarray(res.astype(np.uint8)).convert('RGB')
res.save(results.joinpath('rdj_2bpp_fs.jpg'))
print()

res = floyd_steinberg_dithering(KYUBEY,2)
res = Image.fromarray(res.astype(np.uint8)).convert('RGB')
res.save(results.joinpath('kyubey_2bpp_fs.jpg'))
print()

#FS dithering to 3 bits/pixel
res = floyd_steinberg_dithering(RDJ,3)
res = Image.fromarray(res.astype(np.uint8)).convert('RGB')
res.save(results.joinpath('rdj_3bpp_fs.jpg'))
print()

res = floyd_steinberg_dithering(KYUBEY,3)
res = Image.fromarray(res.astype(np.uint8)).convert('RGB')
res.save(results.joinpath('kyubey_3bpp_fs.jpg'))
print()

