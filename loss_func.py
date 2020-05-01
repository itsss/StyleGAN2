import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as torchF
import torch
import numpy as np

from torch.autograd import Variable
from torch.autograd import grad

# R1 and R2 regularizers
def D_logistic_r1(real_image, Discriminator, gamma=10.0):
    reals = Variable(real_image, requires_grad=True).to(real_image.device)
    real_logit = Discriminator(reals)
    real_grads = grad(torch.sum(real_logit), reals)[0]
    gradient_pen = torch.sum(torch.mul(real_grads, real_grads), dim=[1,2,3])
    return gradient_pen * (gamma * 0.5)

def D_logistic_r2(fake_image, real_image, Discriminator, gamma=10.0):
    real_image = Variable(real_image, requires_grad=True).to(real_image.device)
    fake_image = Variable(fake_image, requires_grad=True).to(fake_image.device)
    real_score = Discriminator(real_image)
    fake_score = Discriminator(fake_image)

    D_logistic_r2_loss = torchF.softplus(fake_score)
    D_logistic_r2_loss = D_logistic_r2_loss + torchF.softplus(-real_score)

    fake_grad = grad(torch.sum(fake_score), fake_image)[0]
    # gradient penalty 부여
    grad_penalty = torch.sum(torch.square(fake_grad), dim=[1,2,3])
    reg = grad_penalty * (gamma * 0.5)
    return D_logistic_r2_loss + reg

# Non-saturating logistic loss with path length regularizer
def G_logistic_ns_pathreg(x, Discriminator, opts, p1_decay = 0.01, p1_weight = 2.0):
    fake_image_out, fake_dlatent_out = x
    fake_image_out = Variable(fake_image_out, requires_grad=True).to(fake_image_out.device)
    fake_score_out = Discriminator(fake_image_out)
    loss_func = torchF.softplus(-fake_score_out)

    fake_dlatent_out = Variable(fake_dlatent_out, requires_grad=True).to(fake_image_out.device)
    p1_noise_vec = torch.randn(fake_image_out.shape) / np.sqrt(fake_image_out.shape[2] * fake_image_out.shape[3])
    p1_noise_vec = p1_noise_vec.to(fake_image_out.device)
    p1_grads = grad(torch.sum(fake_image_out * p1_noise_vec), fake_dlatent_out, retain_graph=True)[0]
    p1_len = torch.sqrt(torch.sum(torch.sum(torch.mul(p1_grads, p1_grads), dim=2), dim=1))
    p1_mean = p1_decay * torch.sum(p1_len)
    p1_penalty = torch.mul(p1_len - p1_mean, p1_len - p1_mean)
    reg = p1_penalty * p1_weight
    return loss_func+reg
