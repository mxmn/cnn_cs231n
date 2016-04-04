import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N, D, C = X.shape[0], X.shape[1], W.shape[1]

  for i in xrange(N):
    scores = X[i].dot(W) # f(j)
    scores -= scores.max()
    det = np.sum(np.exp(scores))
    loss += np.log(det)  - scores[y[i]]

    dW[:,y[i]] -= X[i] # W[:,y[i]]
    for j in xrange(C):
      dW[:,j] += np.exp(scores[j]) /det * X[i]

  loss /= N
  dW /= N

  # regularization
  loss += 0.5 * reg * np.sum(W*W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N, D, C = X.shape[0], X.shape[1], W.shape[1]
  S = X.dot(W)
  S -= S.max()
  denom = np.sum(np.exp(S), axis=1)
  loss = np.mean(np.log(denom) - S[range(N),y]) + 0.5*reg*np.sum(W*W)
  U = np.exp(S) / np.repeat(denom, C).reshape((N,C))
  Q = np.zeros((N,C)); Q[range(N),y] = 1;
  dW = X.T.dot(U-Q) / N  + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW
