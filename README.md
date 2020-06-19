Hand Written Digit Recognition using Multi-Layer Neural Network
===============================================================


Introduction
------------

Hand written digit recognition using multi-layer neural network build using only python's numpy library. The included pre-trained model achieved an accuracy of 91.79 % on MNIST dataset.

## Requirements 

- numpy
- mnist

## Usage

- train a model
	> python recognizer.py --train model_name

- Continue training from a specific model
	> python recognizer.py --train model_name --loadmodel old_model_name

- Evaluate pre-trained model
	> python recognizer.py --evaluate trained_weights.npz