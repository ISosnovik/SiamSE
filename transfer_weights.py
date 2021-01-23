"""
MIT License

Copyright (c) 2021 Ivan Sosnovik, Artem Moskalev
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
from lib.models.models import SESiamFCResNet22

# This script is used to transfer the weights from a pretrained standard model to its scale-equivariant counterpart.
# We transfered the weights from the models released by Zhipeng Zhang, Houwen Peng
# Code: https://github.com/researchmm/SiamDW
# You can download the weights we obtained. Read the README file for the instructions.
# If you want to generate your own weights, just set these variables according to your case.

MODEL_ARCH = SESiamFCResNet22
WEIGHTS_SRC_PATH = 'CIResNet22_PRETRAIN.model'
OUTPUT_FILENAME = 'SESiamFCResNet22_pretrained.pth'


def tikhonov_reg_lstsq(A, B, eps=1e-12):
    '''|A x - B| + |Gx| -> min_x
    '''
    W = A.shape[1]
    A_inv = np.linalg.inv(A.T @ A + eps * np.eye(W)) @ A.T
    return A_inv @ B


def copy_state_dict_bn(dict_target, dict_origin, key_target, key_origin):
    for postfix in ['weight', 'bias', 'running_mean', 'running_var']:
        dict_target[key_target + '.' + postfix] = dict_origin[key_origin + '.' + postfix]


def copy_state_dict_conv_hh_1x1(dict_target, dict_origin, key_target, key_origin):
    dict_target[key_target + '.weight'] = dict_origin[key_origin + '.weight']
    if key_target + '.bias' in dict_target:
        assert key_origin + '.bias' in dict_origin
        dict_target[key_target + '.bias'] = dict_origin[key_origin + '.bias']


def copy_state_dict_conv_hh_1x1_interscale(dict_target, dict_origin, key_target, key_origin):
    w_original = dict_target[key_target + '.weight']
    w_original *= 0
    w_original[:, :, 0] = dict_origin[key_origin + '.weight']
    dict_target[key_target + '.weight'] = w_original
    if key_target + '.bias' in dict_target:
        assert key_origin + '.bias' in dict_origin
        dict_target[key_target + '.bias'] = dict_origin[key_origin + '.bias']


def copy_state_dict_conv_zh(dict_target, dict_origin, key_target, key_origin, scale=0, eps=1e-12):
    weight = dict_origin[key_origin + '.weight']
    basis = dict_target[key_target + '.basis'][:, scale]
    dict_target[key_target + '.weight'] = _approximate_weight(basis, weight, eps)

    if key_target + '.bias' in dict_target:
        assert key_origin + '.bias' in dict_origin
        dict_target[key_target + '.bias'] = dict_origin[key_origin + '.bias']


def copy_state_dict_conv_hh(dict_target, dict_origin, key_target, key_origin, scale=0, eps=1e-12):
    weight = dict_origin[key_origin + '.weight']
    basis = dict_target[key_target + '.basis'][:, scale]
    x = torch.zeros_like(dict_target[key_target + '.weight'])
    x[:, :, 0] = _approximate_weight(basis, weight, eps)

    dict_target[key_target + '.weight'] = x

    if key_target + '.bias' in dict_target:
        assert key_origin + '.bias' in dict_origin
        dict_target[key_target + '.bias'] = dict_origin[key_origin + '.bias']


def _approximate_weight(basis, target_weight, eps=1e-12):
    C_out, C_in, h, w = target_weight.shape
    B, H, W = basis.shape
    with torch.no_grad():
        basis = F.pad(basis, [(w - W) // 2, (w - W) // 2, (h - W) // 2, (h - H) // 2])
        target_weight = target_weight.view(C_out * C_in, h * w).detach().numpy()
        basis = basis.reshape(B, h * w).detach().numpy()

    assert basis.shape[0] == basis.shape[1]

    matrix_rank = np.linalg.matrix_rank(basis)

    if matrix_rank == basis.shape[0]:
        x = np.linalg.solve(basis.T, target_weight.T).T
    else:
        print('  !!! basis is incomplete. rank={} < num_funcs={}. '
              'weights are approximateb by using '
              'tikhonov regularization'.format(matrix_rank, basis.shape[0]))
        x = tikhonov_reg_lstsq(basis.T, target_weight.T, eps=eps).T

    diff = np.linalg.norm(x @ basis - target_weight)
    norm = np.linalg.norm(weight) + 1e-12
    print('  relative_diff={:.1e}, abs_diff={:.1e}'.format(diff / norm, diff))
    x = torch.Tensor(x)
    x = x.view(C_out, C_in, B)
    return x


def convert_param_name_to_layer(name):
    layer_name = '.'.join(name.split('.')[:-1])
    if 'bn' in name:
        return layer_name, 'bn'
    if 'conv' in name:
        return layer_name, 'conv'
    if 'downsample.0' in name:
        return layer_name, 'conv'
    if 'downsample.1' in name:
        return layer_name, 'bn'
    if name == 'features.mean' or name == 'features.std':
        return layer_name, 'save'
    print(name)
    raise NotImplementedError


model = MODEL_ARCH().eval()

src_state_dict = torch.load(WEIGHTS_SRC_PATH)
dest_state_dict = model.state_dict()


keys = list(dest_state_dict.keys())
layers = list(set(['.'.join(key.split('.')[:-1]) for key in keys]))
layers_repr = [convert_param_name_to_layer(name) for name in keys]
layers_repr = list(set(layers_repr))
for layer_name, layer_type in layers_repr:
    print('Layer {}:'.format(layer_name))

    if layer_type == 'bn':
        copy_state_dict_bn(dest_state_dict, src_state_dict, layer_name, layer_name)

    if layer_type == 'conv':
        weight = dest_state_dict[layer_name + '.weight']
        if weight.shape[-1] == weight.shape[-2] == 1:
            if len(weight.shape) == 4:
                copy_state_dict_conv_hh_1x1(dest_state_dict, src_state_dict, layer_name, layer_name)
            elif len(weight.shape) == 5:
                copy_state_dict_conv_hh_1x1_interscale(
                    dest_state_dict, src_state_dict, layer_name, layer_name)
            else:
                raise NotImplementedError
        elif len(weight.shape) == 4:
            copy_state_dict_conv_hh(dest_state_dict, src_state_dict, layer_name, layer_name)
        else:
            copy_state_dict_conv_zh(dest_state_dict, src_state_dict, layer_name, layer_name)
    if layer_type == 'save':
        pass

if not os.path.exists('./pretrain'):
    os.makedirs('./pretrain')
output_path = os.path.join('./pretrain', OUTPUT_FILENAME)
torch.save(dest_state_dict, output_path)
print()
print('Model is saved as "{}"'.format(output_path))
