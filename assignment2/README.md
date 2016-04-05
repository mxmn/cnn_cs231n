# Assignment 2

## Fully-connected Neural Network
- Modular network layers implementation
- _Affine_ layer
- _ReLU_ layer
- _Sandwich_ layers: common patterns of combined layers
- _Loss function_ layers: Softmax and SVM
- _Solver_
- Visualization of training loss, and train and test accuracy over the
  iterations and epochs.
- __Update rules__ for the learning rate
  - SGD + Momentum
  - RMSProp: similar to Adam, uses per-parameter learning rate updates
    based on a running average of the second moments of gradients.
  - Adam:
    - Adam performed better on training, test, and validation sets
      than RMSProp.

## Batch Normalization Layer
- Machine learning tends to work better when the input data consists
  of uncorrelated features. While one could normalize (scale,
  variance) the inputs, the activations at deeper layers might still
  become correlated. The batch normalization layers are intended to
  decorrelate such layers. However, in some networks features with
  non-zero mean and non-unit variance might be preferrable. To this
  end, the batch normalization layer includes learnable _shift_ and
  _scale_ parameters for each feature dimension.


## Dropout Layer
- Helps with regularization a network by setting during the forward
  pass some randomly selected features to zero.


## Convolutional Networks
- Convolutional layers
- Max pooling
- Convolutional _sandwich_ layers: `conv_relu_pool`, `conv_relu`.
- Sanity checks:
  - loss, gradient check
  - overfit small data
- *TODO* Spatial Batch Normalization
