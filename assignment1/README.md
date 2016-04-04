# Assignment 1

## kNN Classifier

- Easy warm-up
- CIFAR-10 data
- Cross-validation

## SVM Classifier

- Linear SVM analytical gradient computation
- Stochastic Gradient Descent (GSD) optimization
- Tuning learning rate
- Regularization
- Loss function
- Numerical gradient check

## Softmax

- Vectorized loss function for the Softmax classifier
- Its analytic gradient

## Two-layer Neural Net

- Two-layer fully connected network
- Forward pass: scores and loss
- Backward pass: gradient of the loss
- Experimentation / fine tuning the hyperparameters
  - One observation: when selectively starting with the weights from a
    previous trial, one can achieve better results in the long term;
    these weights can be slightly randomized as well.

## Image Features Extraction

- Histogram of Oriented Gradients (HOG) features
- Color histogram using the hue channel in HSV color space
- Multiclass SVM on top of the extracted features
  - performs better than when training directly on raw pixels
- Using Neural Nets with the extracted features is expected to perform even better
