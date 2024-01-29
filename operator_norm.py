import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn

def find_input_shapes(model,device):

    '''
    return the list of the input shape of each convolutional layer of the model
    '''

    input_shapes = []
    def hook_fn(module, input, output):
        input_shapes.append(input[0].shape)
    for layer in model.children():
        if isinstance(layer, nn.Conv2d):  
            layer.register_forward_hook(hook_fn)

    device = device  if next(model.parameters()).is_cuda else 'cpu'
    try :
        
        dummy_input = torch.randn(1, 1, 224, 224).to(device)
        _ = model(dummy_input)
    except:
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        _ = model(dummy_input)

    return input_shapes


def prep_params(model, device = 'cuda'):

    '''
    return a dictionnary with the parameters of model and the input shape of each layer
    '''
    res = model.state_dict()
    feature_shapes = find_input_shapes(model, device)
    for i, shape in enumerate(feature_shapes):
        res[f'input_shape{i+1}'] = shape
    return res


def singular_values(kernel,input_shape):  

    '''
    compute the singular values using the method presented in https://arxiv.org/pdf/1805.10408.pdf
    '''
    transforms=np.fft.fft2(kernel,input_shape, axes =[0,1]) 
    return np.linalg.svd(transforms,compute_uv=False) 


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



def Clip_OperatorNorm_NP(filter, inp_shape, clip_to):
  # compute the singular values using FFT
  # first compute the transforms for each pair of input and output channels
  transform_coeff = np.fft.fft2(filter, inp_shape, axes=[0, 1])

  # now, for each transform coefficient, compute the singular values of the
  # matrix obtained by selecting that coefficient for
  # input-channel/output-channel pairs
  U, D, V = np.linalg.svd(transform_coeff, compute_uv=True, full_matrices=False)
  D_clipped = np.minimum(D, clip_to)
  if filter.shape[2] > filter.shape[3]:
    clipped_transform_coeff = np.matmul(U, D_clipped[..., None] * V)
  else:
    clipped_transform_coeff = np.matmul(U * D_clipped[..., None, :], V)
  clipped_filter = np.fft.ifft2(clipped_transform_coeff, axes=[0, 1]).real
  args = [range(d) for d in filter.shape]
  print(args)
  return clipped_filter.reshape(filter.shape)#[np.ix_(*args)]


def project_model(model, inp_shape, clip_to):
    with torch.no_grad():
        for i,layer in enumerate(model.features):
            if isinstance(layer, nn.Conv2d):
                weights = layer.weight.data
                clipped_weights = Clip_OperatorNorm_NP(weights, inp_shape, clip_to)
                model.features[i].weight.data = torch.tensor(clipped_weights)
            elif isinstance(layer, nn.Linear):
                weights = layer.weight.data
                clipped_weights = Clip_OperatorNorm_NP(weights, weights[-2:].shape, clip_to)
                model.features[i].weight.data = torch.tensor(clipped_weights)
            
        

