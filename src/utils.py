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

def load_prep_image(filepath, target_size=None):
    img = misc.imread(filepath)
    if(target_size != None):
        if(img.shape[:2] != target_size):
            img = misc.imresize(img, target_size)
    img = img[..., ::-1].astype('float32') #RGB -> BGR
    img = sub_imagenet_mean(img)
    img = np.expand_dims(img, axis=0)
    return img

def save_deprep_image(filepath, img):
    img = np.squeeze(img)
    img = add_imagenet_mean(img)
    img = img[..., ::-1].astype('uint8')
    misc.imsave(filepath, img)
