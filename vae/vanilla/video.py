import cv2
import os
import imageio

image_folder = 'vae/generations'


images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

images2 = []
for filename in images:
    images2.append(imageio.imread(os.path.join(image_folder,filename)))
imageio.mimsave('vae/movie.gif', images2)