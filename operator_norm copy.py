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


def compute_operator_distance(dict1,dict2,input_shape,n_conv = 20): 

    '''
    dict1 = output of prep_params(model1)
    return the operator norm of K1-K2, where K1 corresponds to the kernels of the first model
    '''
    operator_norm = 0 
    n_param = 0
    for i in range(0, n_conv+1):
        if f'features.{i}.weight' in dict1:
            k1,k2 = dict1[f'features.{i}.weight'].cpu(), dict2[f'features.{i}.weight'].cpu()
            #size = dict1[f'input_shape{i}'][-2:]

            operator_norm += singular_values(k1-k2, input_shape).max()
            n_param+= np.prod(k1.shape)
    for i in (0,4):
        if f'classifier.{i}.weight' in dict1:
            k1,k2 = dict1[f'classifier.{i}.weight'].cpu(), dict2[f'classifier.{i}.weight'].cpu()

            operator_norm += singular_values(k1-k2, input_shape=k1.shape[-2:]).max()
            n_param+= np.prod(k1.shape)

    return operator_norm, n_param



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
        


    
# def project_model(model, inp_shape, clip_to):
#     with torch.no_grad():
#         for i,layer in enumerate(model.features):
#             if isinstance(layer, nn.Conv2d):
#                 weights = layer.weight.data
#                 clipped_weights = clip_kernel_layer(weights, inp_shape, clip_to)
#                 model.features[i].weight.data = torch.tensor(clipped_weights)
#             elif isinstance(layer, nn.Linear):
#                 weights = layer.weight.data
#                 clipped_weights = Clip_OperatorNorm_NP(weights, weights[-2:].shape, clip_to)
#                 model.features[i].weight.data = torch.tensor(clipped_weights)


# def project_model(model1, model2, inp_shape):
#     operator_norm = 0 
#     n_param = 0
#     with torch.no_grad():
#         for i,layer in enumerate(model.features):
#             if isinstance(layer, nn.Conv2d):
#                 weights = layer.weight.data
#                 clipped_weights = Clip_OperatorNorm_NP(weights, inp_shape, clip_to)
#                 model.features[i].weight.data = torch.tensor(clipped_weights)
#             elif isinstance(layer, nn.Linear):
#                 weights = layer.weight.data
#                 clipped_weights = Clip_OperatorNorm_NP(weights, weights[-2:].shape, clip_to)
#                 model.features[i].weight.data = torch.tensor(clipped_weights)


# def modify_model_weights(model, inp_shape, clip_to):
#     """
#     Apply a function to the weights of all layers in a PyTorch model.

#     Parameters:
#     model (nn.Module): The model whose weights will be modified.
#     f (function): A function that takes a weight tensor as input and returns a modified weight tensor.
#     """
#     with torch.no_grad():
#         for name, param in model.named_parameters():
#             if 'weight' in name:
#                 # Get the layer from the name
#                 layer_name = name.split('.')[0]

#                 # Check if the layer is a convolutional layer
#                 if isinstance(getattr(model, layer_name), nn.Conv2d) :
#                     param.copy_(clip_kernel_layer(param, inp_shape, clip_to))
               
                
#                 if isinstance(getattr(model, layer_name), nn.Linear) :
#                     param.copy_(clip_linear_layer(param, param[-2:].shape, clip_to))
#                 # Apply the function f to the parameter
                
            
            
        

