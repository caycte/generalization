import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn






def singular_values(kernel,input_shape=None,conv=True):  

    '''
    compute the singular values using the method presented in https://arxiv.org/pdf/1805.10408.pdf
    '''
    if conv : 
        transforms=np.fft.fft2(kernel,input_shape, axes =[0,1]) 
        return np.linalg.svd(transforms,compute_uv=False) 
    else : 
        return np.linalg.svd(kernel,compute_uv=False) 


def clip_kernel_layer(kernel, inp_shape, clip_to):
  # compute the singular values using FFT
  # first compute the transforms for each pair of input and output channels
  clip = True
  while clip:
    transform_coeff = np.fft.fft2(kernel, inp_shape, axes=[0, 1])

    # now, for each transform coefficient, compute the singular values of the
    # matrix obtained by selecting that coefficient for
    # input-channel/output-channel pairs
    U, D, V = np.linalg.svd(transform_coeff, compute_uv=True, full_matrices=False)
    D_clipped = np.minimum(D, clip_to)
    if kernel.shape[2] > kernel.shape[3]:
        clipped_transform_coeff = np.matmul(U, D_clipped[..., None] * V)
    else:
        clipped_transform_coeff = np.matmul(U * D_clipped[..., None, :], V)
    clipped_kernel = np.fft.ifft2(clipped_transform_coeff, axes=[0, 1]).real
    args = [range(d) for d in kernel.shape]
    kernel = torch.tensor(clipped_kernel[np.ix_(*args)])
    
    s_value = singular_values(kernel, inp_shape).max()
    
    if s_value<1 + 1e-3:
        clip =False
  return kernel


def clip_linear_layer(weight, clip_to):
 
  U,S,Vh = np.linalg.svd(weight, compute_uv=True, full_matrices=False)
  S = S/np.abs(S).max()
  return  torch.tensor(np.dot(U * S, Vh))


def clip_model_weights(model, inp_shape, clip_to=1):
    """
    Apply a function to the weights of all layers in a PyTorch model.

    Parameters:
    model (nn.Module): The model whose weights will be modified.
    f (function): A function that takes a weight tensor as input and returns a modified weight tensor.
    """
    with torch.no_grad():
        for layer in model.conv:
            
            if isinstance(layer, nn.Conv2d) :
                layer.weight.data.copy_(clip_kernel_layer(layer.weight.data, inp_shape, clip_to))
               
        for layer in model.classifier:       
            if isinstance(layer, nn.Linear) :
             
                layer.weight.data.copy_(clip_linear_layer(layer.weight.data, clip_to))
                # Apply the function f to the parameter
 

def metrics(model1,model2, input_shape):
    operator_norm,n_param = 0,0
    for layer1,layer2 in zip(model1.conv, model2.conv):
        if isinstance(layer1, nn.Conv2d):

            kernel = layer1.weight.data-layer2.weight.data
            operator_norm+= singular_values(kernel,input_shape).max()
            n_param += np.prod(kernel.shape)
    for layer1, layer2 in zip(model1.classifier, model2.classifier):
        if isinstance(layer1, nn.Linear):
            kernel = layer1.weight.data-layer2.weight.data
            operator_norm+= singular_values(kernel,conv=False).max()
            n_param += np.prod(kernel.shape)

    return operator_norm,n_param
        

