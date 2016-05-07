# Assignment 3

## Recurrent Neural Networks (RNNs)
- Image captioning with Vanilla RNNs
- MS-COCO dataset
- Single time step, and full layer forward and backward passes.

## Long-Short Term Memory (LSTM) RNNs
- Single time step, and full layer forward and backward passes.

## Image Gradients: Saliency maps and Fooling Images
- pretrained TinyImageNet model
- **Saliency map**: looking at backpropagated signature, one can see which
  spatial location of the image contributes the most to the
  classification of a particular target.
  - One note: the backpropagation is performed on the raw scores
    (weighted in some way by the target class), and not on the
    probabilities.
  - Based on the paper: Karen Simonyan, Andrea Vedaldi, and Andrew
    Zisserman. "Deep Inside Convolutional Networks: Visualising Image
    Classification Models and Saliency Maps", ICLR Workshop 2014.
- **Fooling Images**: after forward and backward propagation with a given
  target class, slightly change the input image by the backpropagated
  dx, so that it becomes closer to the target class. This can go on
  until the image is transformed so far that the network is fooled
  into believing that it is the target class.
  - Based on the paper: Szegedy et al, "Intriguing properties of
    neural networks", ICLR 2014

## Image Generation: Classes, Inversion, and DeepDream
- pretrained TinyImageNet model
- *Class Visualization*: Regularized gradient ascend to create an
  image (from noise) based on class features.
  - Based on:
    - Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. "Deep
      Inside Convolutional Networks: Visualising Image Classification
      Models and Saliency Maps", ICLR Workshop 2014.
    - Yosinski et al, "Understanding Neural Networks Through Deep
      Visualization", ICML 2015 Deep Learning Workshop
- **Feature Inversion**:
