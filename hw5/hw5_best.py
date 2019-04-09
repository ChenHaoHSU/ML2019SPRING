import sys
import numpy as np
import pandas as pd
import random as rand
from PIL import Image

input_dir = sys.argv[1]
output_dir = sys.argv[2]

print('# Input dir : {}'.format(input_dir))
print('# Output dir : {}'.format(output_dir))

def load_input(input_dir):
    input_images = []
    for i in range(200):
        image_file = '{}/{:03d}.png'.format(input_dir, i)
        im = Image.open(image_file)
        input_images.append(im)
    return input_images

def write_output(images, output_dir):
    for i, im in enumerate(images):
        output_file = '{}/{:03d}.png'.format(output_dir, i)
        im.save(output_file)

input_images = load_input(input_dir)
print('# Load {} images'.format(len(input_images)))

rand_list = [-23, 23]
print('# Processing...')
for i, im in enumerate(input_images):
    print(i)
    pixels = im.load()
    width, height = im.size
    for x in range(0, width):
        for y in range(0, height):
            increment = 23 if ((x + y)%6) < 3 else -23
            # increment = 8 if xx == yy else -8
            r = pixels[x, y][0]
            g = pixels[x, y][1]
            b = pixels[x, y][2]

            pixels[x, y] = (int(r + increment), int(g + increment), int(b + increment))
            # pixels[x, y] = (int(r + rand.choice(rand_list)), int(g + rand.choice(rand_list)), int(b + rand.choice(rand_list)))

write_output(input_images, output_dir)
print('# Write {} images'.format(len(input_images)))


