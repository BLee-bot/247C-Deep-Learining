import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from cs231n.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize the weights and biases of a three layer CNN. To initialize:
    #     - the biases should be initialized to zeros.
    #     - the weights should be initialized to a matrix with entries
    #         drawn from a Gaussian distribution with zero mean and 
    #         standard deviation given by weight_scale.
    # ================================================================ #
    C, H, W = input_dim
        
    self.params['W1'] = np.random.randn(num_filters, C, filter_size, filter_size)*weight_scale
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = np.random.randn(num_filters*(H//2)*(W//2), hidden_dim)*weight_scale
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = np.random.randn(hidden_dim, num_classes)*weight_scale
    self.params['b3'] = np.zeros(num_classes)



    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the three layer CNN.  Store the output
    #   scores as the variable "scores".
    # ================================================================ #
    
    out1, cache1 = conv_relu_pool_forward(X, W1, b1,conv_param, pool_param)
    out2, cache2 = affine_relu_forward(out1, W2, b2)
    out3, cache3 = affine_forward(out2, W3, b3)
    scores = out3

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    if y is None:
      return scores
    
    loss, grads = 0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of the three layer CNN.  Store the grads
    #   in the grads dictionary, exactly as before (i.e., the gradient of 
    #   self.params[k] will be grads[k]).  Store the loss as "loss", and
    #   don't forget to add regularization on ALL weight matrices.
    # ================================================================ #

    loss, dscores = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))

    dX3, dW3, db3 = affine_backward(dscores, cache3)
    dX2, dW2, db2 = affine_relu_backward(dX3, cache2)
    dX1, dW1, db1 = conv_relu_pool_backward(dX2, cache1)

    grads['W1'] = dW1 + self.reg * W1
    grads['W2'] = dW2 + self.reg * W2
    grads['W3'] = dW3 + self.reg * W3
    grads['b1'] = db1
    grads['b2'] = db2
    grads['b3'] = db3

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads
  
  
pass

'''
#---------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------#
'''

class ConvNet(object):
  """
  # Designed 6 Layer networks ConvNet
  # 3x3 filter, 2x2 max pool
  # Applied batchnorm after every layers
  # Network archicecture = [conv-relu-norm]-[conv-relu-pool-norm]x3-affine-relu-norm-affine-softmax
  # 1st conv Layer used 8 filters
  # 2nd conv Layer used 16 filters
  # 3rd conv Layer used 32 filters
  # 4th conv Layer used 64 filters
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=8, filter_size=3,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.25,
               dtype=np.float32, use_batchnorm=True):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize the weights and biases of a three layer CNN. To initialize:
    #     - the biases should be initialized to zeros.
    #     - the weights should be initialized to a matrix with entries
    #         drawn from a Gaussian distribution with zero mean and 
    #         standard deviation given by weight_scale.
    # ================================================================ #
    C, H, W = input_dim
        
    self.params['W1'] = np.random.randn(num_filters, C, filter_size, filter_size)*weight_scale
    self.params['b1'] = np.zeros(num_filters)
    self.params['gamma1'] = np.ones(num_filters)
    self.params['beta1'] = np.zeros(num_filters)
    self.params['W2'] = np.random.randn(num_filters*2,num_filters,filter_size,filter_size)*weight_scale
    self.params['b2'] = np.zeros(num_filters*2)
    self.params['gamma2'] = np.ones(num_filters*2)
    self.params['beta2'] = np.zeros(num_filters*2)
    self.params['W3'] = np.random.randn(num_filters*4,num_filters*2,filter_size,filter_size)*weight_scale
    self.params['b3'] = np.zeros(num_filters*4)
    self.params['gamma3'] = np.ones(num_filters*4)
    self.params['beta3'] = np.zeros(num_filters*4)
    self.params['W4'] = np.random.randn(num_filters*8,num_filters*4,filter_size,filter_size)*weight_scale
    self.params['b4'] = np.zeros(num_filters*8)
    self.params['gamma4'] = np.ones(num_filters*8)
    self.params['beta4'] = np.zeros(num_filters*8)
    self.params['W5'] = np.random.randn(8*num_filters*(H//8)*(W//8), hidden_dim)*weight_scale
    self.params['b5'] = np.zeros(hidden_dim)
    self.params['gamma5'] = np.ones(hidden_dim)
    self.params['beta5'] = np.zeros(hidden_dim)
    self.params['W6'] = np.random.randn(hidden_dim, num_classes)*weight_scale
    self.params['b6'] = np.zeros(num_classes)



    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    self.bn_params = []
    self.bn_params = [{'mode': 'train'} for i in np.arange(6)]
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    W6, b6 = self.params['W6'], self.params['b6']

    gamma1, beta1 = self.params['gamma1'], self.params['beta1']
    gamma2, beta2 = self.params['gamma2'], self.params['beta2']
    gamma3, beta3 = self.params['gamma3'], self.params['beta3']
    gamma4, beta4 = self.params['gamma4'], self.params['beta4']
    gamma5, beta5 = self.params['gamma5'], self.params['beta5'] 

    mode = 'test' if y is None else 'train'

    for bn_param in self.bn_params:
      bn_param[mode] = mode
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the three layer CNN.  Store the output
    #   scores as the variable "scores".
    # ================================================================ #
    out1, cache1 = conv_relu_forward(X, W1, b1, conv_param)
    out1, bncache1 = spatial_batchnorm_forward(out1, gamma1, beta1, self.bn_params[1])
    out2, cache2 = conv_relu_pool_forward(out1, W2, b2, conv_param, pool_param)
    out2, bncache2 = spatial_batchnorm_forward(out2, gamma2, beta2, self.bn_params[2])
    out3, cache3 = conv_relu_pool_forward(out2, W3, b3, conv_param, pool_param)
    out3, bncache3 = spatial_batchnorm_forward(out3, gamma3, beta3, self.bn_params[3])
    out4, cache4 = conv_relu_pool_forward(out3, W4, b4, conv_param, pool_param)
    out4, bncache4 = spatial_batchnorm_forward(out4, gamma4, beta4, self.bn_params[4])
    out5, cache5 = affine_relu_forward(out4, W5, b5)
    out5, bncache5 = batchnorm_forward(out5, gamma5, beta5, self.bn_params[5])
    out6, cache6 = affine_forward(out5, W6, b6)
    scores = out6
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    if y is None:
      return scores
    
    loss, grads = 0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of the three layer CNN.  Store the grads
    #   in the grads dictionary, exactly as before (i.e., the gradient of 
    #   self.params[k] will be grads[k]).  Store the loss as "loss", and
    #   don't forget to add regularization on ALL weight matrices.
    # ================================================================ #

    loss, dscores = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3)+np.sum(W6*W6) + np.sum(W4*W4) + np.sum(W5*W5))

    dX6, dW6, db6 = affine_backward(dscores, cache6)
    dX6, dgamma5, dbeta5 = batchnorm_backward(dX6, bncache5)
    dX5, dW5, db5 = affine_relu_backward(dX6, cache5)
    dX5, dgamma4, dbeta4 = spatial_batchnorm_backward(dX5, bncache4)
    dX4, dW4, db4 = conv_relu_pool_backward(dX5, cache4)
    dX4, dgamma3, dbeta3 = spatial_batchnorm_backward(dX4, bncache3)
    dX3, dW3, db3 = conv_relu_pool_backward(dX4, cache3)
    dX3, dgamma2, dbeta2 = spatial_batchnorm_backward(dX3, bncache2)
    dX2, dW2, db2 = conv_relu_pool_backward(dX3, cache2)
    dX2, dgamma1, dbeta1 = spatial_batchnorm_backward(dX2, bncache1)
    dX1, dW1, db1 = conv_relu_backward(dX2, cache1)
    grads['W1'] = dW1 + self.reg * W1
    grads['W2'] = dW2 + self.reg * W2
    grads['W3'] = dW3 + self.reg * W3
    grads['W4'] = dW4 + self.reg * W4
    grads['W5'] = dW5 + self.reg * W5
    grads['W6'] = dW6 + self.reg * W6
    grads['b1'] = db1
    grads['b2'] = db2
    grads['b3'] = db3
    grads['b4'] = db4
    grads['b5'] = db5
    grads['b6'] = db6
    grads['gamma1'], grads['beta1'] = dgamma1, dbeta1
    grads['gamma2'], grads['beta2'] = dgamma2, dbeta2
    grads['gamma3'], grads['beta3'] = dgamma3, dbeta3
    grads['gamma4'], grads['beta4'] = dgamma4, dbeta4
    grads['gamma5'], grads['beta5'] = dgamma5, dbeta5


    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads
  
  
pass
