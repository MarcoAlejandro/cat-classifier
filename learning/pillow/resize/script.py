"""
    Resize of an image using Pillow thumbnail and resize functions.

    thumbnail() will preserve the aspect ratio.
    resize() will force the pixels into a new shape. This implies the use of
        so called "resampling algorithms".
"""
from PIL import Image

image = Image.open("../images/opera_house.jpg")
print(image.size)
image.thumbnail((100, 100))
print(image.size)
image_rs = image.resize((200, 200))
print(image_rs.size)
