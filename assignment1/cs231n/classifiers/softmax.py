import numpy as np
from random import shuffle
from past.builtins import xrange

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

  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
      scores = X[i].dot(W)
      sum_exp = np.sum(np.exp(scores))
      soft_max = np.exp(scores) / sum_exp
      loss += -np.log(soft_max[y[i]])

      dW += X[i].reshape((-1,1)).dot(soft_max.reshape((1,-1)))
      dW[:, y[i]] -= X[i]


  loss = loss / num_train + reg * np.sum(W * W)
  dW = dW / num_train + 2 * reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W)
  scores_correct = scores[range(num_train), y]

  sum_exp = np.sum(np.exp(scores), axis=1)

  soft_max = np.log(np.exp(scores_correct) / sum_exp)

  loss = -np.mean(soft_max) + reg * np.sum(W*W)

  soft_max_matrix = np.exp(scores) / sum_exp.reshape((-1,1))
  soft_max_matrix[range(num_train), y] -= 1
  dW = X.transpose().dot(soft_max_matrix) / num_train + reg*2*W



  return loss, dW

