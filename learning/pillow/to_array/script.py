"""
    Loads an image using Pillow, and coverts it to numpy array.

    The images can also be loaded using matplotlib
"""
from PIL import Image
from numpy import asarray
from matplotlib import (
    image,
    pyplot
)

# Using pillow
im = Image.open('../images/opera_house.jpg')
data_im1 = asarray(im)
print(data_im1.shape)
im2 = Image.fromarray(data_im1)
print(im2.format)
print(im2.mode)
print(im2.size)

# Using matplotlib
data_mpl = image.imread('../images/opera_house.jpg')
print(data_mpl.dtype)
print(data_mpl.shape)

assert data_im1.shape == data_mpl.shape
