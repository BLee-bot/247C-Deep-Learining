import numpy as np
from nndl.layers import *
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

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
  npad = ((0, 0), (0, 0), (pad, pad), (pad, pad)) #(N, C, H, W)
  xpad = np.pad(x, pad_width=npad, mode='constant', constant_values=0)#(N, C, H+2pad, W+2pad)
  out = np.zeros((x.shape[0],w.shape[0],1 + (x.shape[2] + 2 * pad - w.shape[2]) // stride,
               1 + (x.shape[3] + 2 * pad - w.shape[3]) // stride))
  for n in range(xpad.shape[0]): #N
    xpadt = xpad[n]
    for f in range(w.shape[0]): #F
      ii, jj = -1, -1
      for i in range(0,xpad.shape[2]-w.shape[2]+1, stride):
        ii += 1
        for j in range(0,xpad.shape[3]-w.shape[3]+1, stride):
          jj += 1
          out[n,f,ii,jj] = np.sum(xpadt[:, i:i+w.shape[2], j:j+w.shape[3]] * w[f]) + b[f]
        jj = -1
      
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  N,C,H,W = x.shape
  dx = np.zeros(x.shape) #(N, C, H, W)
  dxpad = np.zeros(xpad.shape)
  st=stride
  for n in range(N): # N
    for f in range(F): # F
      for ho in range(out_height): # HO
        for wo in range(out_width): # WO
          dxpad[n,:,ho*st:ho*st+f_height,wo*st:wo*st+f_width]+=w[f]*dout[n,f,ho,wo]
  dx = dxpad[:,:,pad:pad+H,pad:pad+W]

  F,C,HH,WW = w.shape
  dw = np.zeros(w.shape) #(F, C, HH, WW)
  for f in range(F): #F
    for c in range(C): #C
      for hh in range(HH): #HH
        for ww in range(WW): #WW
          dw[f,c,hh,ww] = np.sum(dout[:,f,:,:]*
                                 xpad[:,c,hh:hh+out_height*st:st,ww:ww+out_width*st:st])

  db = np.zeros(b.shape)
  for f in range(F):
    db[f] = np.sum(dout[:,f,:,:])

  
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  H,W = x.shape[2], x.shape[3]
  N,C = x.shape[0], x.shape[1]
  ph=pool_param['pool_height']
  pw=pool_param['pool_width']
  st=pool_param['stride']
  out = np.zeros((N,C,1+(H-ph)//st,1+(W-pw)//st))

  for n in range(N):
    for c in range(C):
      for h in range(H//st):
        for w in range(W//st):
          xtemp = x[n,c,h*st,w*st]
          for hh in range(ph):
            for ww in range(pw):
              if xtemp <= x[n,c,h*st+hh,w*st+ww]:
                xtemp = x[n,c,h*st+hh,w*st+ww]
          out[n,c,h,w]=xtemp
          
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  H,W = x.shape[2], x.shape[3]
  N,C = x.shape[0], x.shape[1]
  ph = pool_height #2
  pw = pool_width #2
  st = stride #2

  dx = np.zeros((N, C, H, W))
  ho = 1+(H-ph)//st
  wo = 1+(W-pw)//st
  for n in range(N):
    for c in range(C):
      for h in range(1+(H-ph)//st):
        for w in range(1+(W-pw)//st):
          xtemp = x[n,c,h*st:h*st+ph,w*st:w*st+pw]
          i,j=np.unravel_index(xtemp.argmax(), xtemp.shape)
          dx[n,c,st*h+i,st*w+j] = dout[n,c,h,w]
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N, C, H, W = x.shape
  
  x = x.transpose(0,2,3,1).reshape(N*H*W,C)

  out, cache = batchnorm_forward(x, gamma, beta, bn_param)

  out = out.reshape(N,H,W,C).transpose(0,3,1,2)

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N, C, H, W = dout.shape

  dout = dout.transpose(0,2,3,1).reshape(N*H*W,C)

  dx, dgamma, dbeta = batchnorm_backward(dout, cache)

  dx = dx.reshape(N,H,W,C).transpose(0,3,1,2)
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta
