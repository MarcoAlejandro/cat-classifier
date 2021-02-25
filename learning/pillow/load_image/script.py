"""
    Open an image using the Pillow Image class, and prints out some information
    about the Image instance.

    Run the interpreter at this folder. Otherwise, the code would not find the image
    relative path.
"""

from PIL import Image

image = Image.open('../images/opera_house.jpg')
print(image.format)
print(image.mode)
print(image.size)
image.show()
