import torch
from inspect import isfunction

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def noise_like(shape, device, repeat=False):
    def repeat_noise(): return torch.randn(
        (1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))

    def noise(): return torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def total_variation_loss(img):
    bs_img, c_img, h_img, w_img = img.size()

    # norm
    norm_img = (img - img.view(bs_img, -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1))
    norm_img = norm_img / torch.max(norm_img.view(bs_img, -1), dim=-1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1) # 0 - 1
    norm_img = norm_img.detach()
    # weight
    avg = (norm_img[:,:,1:,:] + norm_img[:,:,:-1,:]) / 2.
    weight_h = torch.pow((1. - avg), 2)

    avg = (norm_img[:,:,:,1:] + norm_img[:,:,:,:-1]) / 2.
    weight_w = torch.pow((1. - avg), 2)

     
    tv_h = (weight_h * torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2)).sum()
    tv_w = (weight_w * torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2)).sum()
    return torch.sqrt((tv_h+tv_w))/(bs_img*c_img*h_img*w_img)

def noise2self(target, pred, mask):
    # mask: b, 1, w, h
    # target: b, 1, w, h
    # pred: b, 1, w, h

    loss = (torch.pow(pred - target, 2) * mask).sum(dim=(-1,-2)) / mask.sum(dim=(-1,-2))
    return loss.mean()

def masked_mse(x, y, mask=None):
    err2 = (x - y) ** 2
    if mask is not None:
        return torch.sum(mask * err2) / mask.sum()
    else:
        return err2.mean()

def masked_l1(x, y, mask=None):
    err2 = torch.abs(x - y)
    if mask is not None:
        return torch.sum(mask * err2) / mask.sum()
    else:
        return err2.mean()

def data_augmentation(img, flip_v, flip_h):
    axis = []
    if flip_v:
        axis.append(2)
    if flip_h:
        axis.append(3)
    if len(axis):
        img = torch.flip(img, axis)
    return img

def calculate_matmul_n_times(n_components, mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (1, k, d, d)
    """
    res = torch.zeros(mat_a.shape).double().to(mat_a.device)
    
    for i in range(n_components):
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)
        mat_b_i = mat_b[0, i, :, :].squeeze()
        res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)
    
    return res

def calculate_matmul(mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (n, k, d, 1)
    """
    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)

def flip_denoise(x, denoise_fn, noise_levels, flips=[(False, False)]):
    b, c, w, h = x.shape
    #flips = [(False, False), (True, False), (False, True), (True, True)]
    supports = []
    
    for f in flips:
        supports.append(data_augmentation(x, f[0], f[1]))
    
    x_recon = denoise_fn(torch.cat(supports, dim=0), noise_levels)
    
    split_x_recon = torch.split(x_recon, b, 0)
    
    supports = []
    for idx, f in enumerate(flips):
        supports.append(data_augmentation(split_x_recon[idx], f[0], f[1]).unsqueeze(1))

    x_recon = torch.mean(torch.cat(supports, dim=1), dim=1, keepdim=False)
    return x_recon

def flip_denoise_noise(x, denoise_fn, noise_levels, flips=[(False, False)]):
    b, c, w, h = x.shape
    #flips = [(False, False), (True, False), (False, True), (True, True)]
    supports = []
    
    for f in flips:
        supports.append(data_augmentation(x, f[0], f[1]))
    
    x_recon = denoise_fn(torch.cat(supports, dim=0), noise_levels)
    
    split_x_recon = torch.split(x_recon, b, 0)
    
    supports = []
    for idx, f in enumerate(flips):
        supports.append(data_augmentation(split_x_recon[idx], f[0], f[1]).unsqueeze(1))

    x_recon = torch.mean(torch.cat(supports, dim=1), dim=1, keepdim=False)
    return x_recon
