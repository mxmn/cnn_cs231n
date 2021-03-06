## Assignment 3

### Recurrent Neural Networks (RNNs)
- [RNN Captioning Notebook](RNN_Captioning.ipynb)
- Image captioning with Vanilla RNNs
- MS-COCO dataset
- Single time step, and full layer forward and backward passes.

### Long-Short Term Memory (LSTM) RNNs
- [LSTM Captioning Notebook](LSTM_Captioning.ipynb)
- Single time step, and full layer forward and backward passes.

### Image Gradients: Saliency maps and Fooling Images
- [Image Gradients Notebook](ImageGradients.ipynb)
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

### Image Generation: Classes, Inversion, and DeepDream
- [Image Generation Notebook](ImageGeneration.ipynb)
- pretrained TinyImageNet model
- **Class Visualization**: Regularized gradient ascend to create an
  image (from noise) based on class features.
  - Based on:
    - Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. "Deep
      Inside Convolutional Networks: Visualising Image Classification
      Models and Saliency Maps", ICLR Workshop 2014.
    - Yosinski et al, "Understanding Neural Networks Through Deep
      Visualization", ICML 2015 Deep Learning Workshop
- **Feature Inversion**: starting with a random noise image, and a
  given set of features at a certain level, iteratively reconstruct
  the image corresponding to these features.
  - The forward and backpropagation is performed only up to the layer
    with the known features.
  - L2 regularization on the image is added to penalize pixels with
    large values.
  - Based on:
    - Aravindh Mahendran, Andrea Vedaldi, "Understanding Deep
      Image Representations by Inverting them", CVPR 2015
    - Yosinski et al, "Understanding Neural Networks Through Deep
      Visualization", ICML 2015 Deep Learning Workshop
- **DeepDream**: fantastika :)
  - Based on Google's
    [DeepDream](https://github.com/google/deepdream/blob/master/dream.ipynb),
    one dreams at selected convolutional layer. The image gets
    propagated to this layer, and for the backpropagation, one
    gradient is set to be the same as the activations of this layer.
  - Check out the deep dreams of the classics by Hopper and Van Gogh!
