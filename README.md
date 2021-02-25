This project contains an image classifier to classify whether a picture is the picture of a cat or not.

This is a self guide project for learning, while taking some courses about deep learning, and implementing 
those concepts on my own environment.

The idea is document a binnacle of the process and find out what went wrong after finish it.  

**This project and the development methodology are by no mean something to imitate in a real world project.**

------

**Development history**

- [Project structure](#project-structure)
- [Input data for the model](#input-data-for-the-model)

-----

## Project structure

- Currently, I've installed python `3.8.0` on my machine, so that's the version I'll use. 
- The project will use `poetry` for virtual environment and dependency management. 
- Project related files will be inside the `project` folder.
- `learning` folder contains files related to learning how to use techlogies down the road.

-----

## Input data for the model. 

Date: 24/02/2021

I need to feed the neural networks using the features vector representations of the images, 
naturally as `numpy` `ndarray` elements. 
I've never represented images using python, so I need to figure out how to do that. 

[This blog entry](https://machinelearningmastery.com/how-to-load-and-manipulate-images-for-deep-learning-in-python-with-pil-pillow/) points out that `Pillow` is the most popular library for simple image manipulation in Python. Let's try to use that.

Examples about reading images as numpy arrays can be found at: `learning/pillow/to_array/script.py` file.

After loading some images, some points of interest arise: 

- All the images for the training model should have the same size and width, which determines the number of features.

Image resize example code can be found at `learning/pillow/resize/script.py`
