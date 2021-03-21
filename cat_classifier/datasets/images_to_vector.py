"""
    This module encapsulate images reading operations required
    in the preprocessing of images.
"""
from PIL import Image
from typing import List
from numpy import asarray, ndarray
import os


def _resize_images(images: List) -> List:
    """
        All images will be resized to a fixed
        64 x 64 px
    """
    return list(
        map(
            lambda i: i.resize((64, 64)),
            images
        )
    )


def _convert_images_to_array(images: List) -> List:
    """
        Turns a list of images into a list
        of numpy arrays
    """
    return list(
        map(
            lambda i: asarray(i),
            images
        )
    )


def _reshape_vectors(array: ndarray) -> ndarray:
    """
        Reshapes the images vector representation:
        (h, w, 3) -> (h * w * 3, 1)

    A trick when you want to flatten a matrix X of shape (a,b,c,d)
    to a matrix X_flatten of shape (b * c * d, a) is to use:
        X.reshape(X.shape[0], -1).T
    """
    return array.reshape(array.shape[0], -1).T


def _get_dataset(relative_path: str) -> ndarray:
    """
        Returns the input vectors for the training images
    """
    images: ndarray
    images_names = []
    dirname = os.path.dirname(__file__)
    training_folder = os.path.join(dirname, relative_path)
    for filename in os.listdir(training_folder):
        if filename.endswith(".jpg"):
            f = os.path.join(training_folder, filename)
            images_names.append(f)

    """
        Apply functional ops: 
        - image objects creation
        - resize of images
        - convert images to array
        - reshape arrays
        
        After transformations, returned data is a numpy array where each 
        "column" is a features vector for an image. 
    """
    images = _reshape_vectors(
        asarray(
            _convert_images_to_array(
                _resize_images(
                    list(
                        map(
                            lambda i: Image.open(i),
                            images_names
                        )
                    )
                )
            )
        )
    )

    return images


def get_training_data() -> ndarray:
    return _get_dataset('../cats/training')


def get_test_data() -> ndarray:
    return _get_dataset('../cats/test')

