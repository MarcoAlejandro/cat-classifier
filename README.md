This project contains an image classifier to classify whether a picture is the picture of a cat or not.

This is a self guide project for learning, while taking some courses about deep learning, and implementing 
those concepts on my own environment.

The idea is document a binnacle of the process and find out what went wrong after finish it.  

**This project and the development methodology are by no mean something to imitate in a real world project.**

------

**Development history**

- [Project structure](#project-structure)
- [Input data for the model](#input-data-for-the-model)
- [Linear regression algorithm](#logistic-regression)

-----

## Project structure

- Currently, I've installed python `3.8.0` on my machine, so that's the version I'll use. 
- The project will use `poetry` for virtual environment and dependency management. 
- Project related files will be inside the `project` folder.
- `learning` folder contains files related to learning how to use techlogies down the road.

-----

## Input data for the model. 

**Date: 24/02/2021**

I need to feed the neural networks using the features vector representations of the images, 
naturally as `numpy` `ndarray` elements. 
I've never represented images using python, so I need to figure out how to do that. 

[This blog entry](https://machinelearningmastery.com/how-to-load-and-manipulate-images-for-deep-learning-in-python-with-pil-pillow/) points out that `Pillow` is the most popular library for simple image manipulation in Python. Let's try to use that.

Examples about reading images as numpy arrays can be found at: `learning/pillow/to_array/script.py` file.

After loading some images, some points of interest arise: 

- All the images for the training model should have the same size and width, which determines the number of features.

Image resize example code can be found at `learning/pillow/resize/script.py`

**Date: 27/02/2021**

To keep things simple, I'll grab some dataset of cat pictures. Download the images, and delegate the 
image treatment code to some component that will run at training time. 

I'll download cat pictures from the cat dataset from Kaggle [https://www.kaggle.com/crawford/cat-dataset]. Not all of 
them, just a few number of pictures ( ~150 ).

Images are in `learning/cats/images.zip`.

**Date: 03/03/2021**

I've added a module (`project.datasets.images_to_vector`) to read the images and turn them into vectorized data for the models.

---------

## Logistic Regression

**Date: 07/03/2021**

The logistic regression is a classification algorithm, and a good first way to gain intuition about the concepts 
used in neural networks such a: 

- Forward and backward propagation
- Gradient descent
- Input weights
- The threshold
- The activation function

`project/models` contains a module with an implementation of logistic regression. 

