from scipy import misc
import numpy as np

def sub_imagenet_mean(img):
    img[..., 0] -= 103.939
    img[..., 1] -= 116.779
    img[..., 2] -= 123.68
    return img

def add_imagenet_mean(img):
    img[..., 0] += 103.939
    img[..., 1] += 116.779
    img[..., 2] += 123.68
    return img

def load_prep_image(filepath):
    img = misc.imread(filepath).astype('float32') 
    img = img[..., ::-1] #RGB -> BGR
    img = sub_imagenet_mean(img)
    return img

def save_deprep_image(filepath, img):
    img = add_imagenet_mean(img)
    img = img[..., ::-1].astype('uint8')
    misc.imsave(filepath, img)
