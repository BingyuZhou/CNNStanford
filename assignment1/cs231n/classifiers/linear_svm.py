import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, y[i]] -= X[i].transpose()
        dW[:, j] += X[i].transpose()

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)



  dW = dW / num_train + 2 * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero


  num_train = X.shape[0]
  scores = X.dot(W)
  correct_class_score = scores[(range(num_train), y)].reshape((num_train, 1))
  margin = scores - correct_class_score + 1
  margin[(range(num_train), y)] = 0
  multiplier = (margin>0).sum(axis=1)
  m = (margin>0)
  dW_j = X.transpose().dot(m)
  n = np.zeros((num_train, W.shape[1]))
  n[[range(y.shape[0]), y]] = multiplier
  dW_y = -X.transpose().dot(n)

  loss_item = margin[margin > 0]
  loss = loss_item.sum()
  loss = loss / num_train + reg * np.sum(W * W)

  dW = (dW_y + dW_j) / num_train + 2*W


  return loss, dW
