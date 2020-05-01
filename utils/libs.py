import torch
import numpy as np

TWO = [pow(2, _) for _ in range(11)]


def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)
    assert k.ndim == 2
    assert k.shape[0] == k.shape[1]
    return k


def _approximate_size(feature_size):
    tmp = map(lambda x: abs(x - int(feature_size)), TWO)
    tmp = list(tmp)

    idxs = tmp.index(min(tmp))
    return pow(2, idxs)


def update_average(model_tgt, model_src, beta):
    def toggle_grad(model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)

    toggle_grad(model_tgt, False)
    toggle_grad(model_src, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

    toggle_grad(model_tgt, True)
    toggle_grad(model_src, True)


class ShrinkFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input_x = input.clone()
        input_x = input_x / torch.max(torch.abs(input_x))
        return input_x

    @staticmethod
    def backward(ctx, grad_output):
        # function
        grad_input = grad_output.clone()
        return grad_input


def weights_init(m):
    classname = m.__class__.__name__
    if classname == 'Conv2d':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


if __name__ == "__main__":
    _approximate_size(0)