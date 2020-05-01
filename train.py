import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms

from torch.autograd import grad

from StyleGAN2 import Generator_stylegan2, Discriminator_stylegan2
from utils.utils import plotLossCurve
from loss_func import D_logistic_r1, D_logistic_r2, G_logistic_ns_pathreg
from opts import TrainOptions, INFO

from torchvision.utils import save_image
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.optim as optim
import numpy as np
import random
import torch
import os

def main(opts):
    # Data load
    loader = sunnerData.DataLoader(sunnerData.ImageDataset(
        root=[[opts.path]],
        transform=transforms.Compose([sunnertransforms.Resize((opts.resolution, opts.resolution)),sunnertransforms.ToTensor(),sunnertransforms.ToFloat(),sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),sunnertransforms.Normalize()])),
        batch_size=opts.batch_size,
        shuffle=True,
        drop_last=True
    )

    # model generation
    start_epoch = 0
    G = Generator_stylegan2(fmap_base=opts.fmap_base,
                    resol=opts.resolution,
                    mapping_layers=opts.mapping_layers,
                    opts=opts,
                    return_dlatents=True)
    D = Discriminator_stylegan2(fmap_base=opts.fmap_base,
                    resol=opts.resolution,
                    structure='resnet')

    # pre-trained weight loading
    if os.path.exists(opts.resume):
        INFO("Load the pre-trained weight!")
        state = torch.load(opts.resume)
        G.load_state_dict(state['G'])
        D.load_state_dict(state['D'])
        start_epoch = state['start_epoch']
    else:
        INFO("pre-trained weight error")

    # multiple GPU support
    if(torch.cuda.device_count() > 1):
        INFO("multiple GPU detected! Total " + str(torch.cuda.device_count()) + '\t GPUs!')
        G = torch.nn.DataParrlel(G)
        D = torch.nn.DataParallel(D)
    G.to(opts.device)
    D.to(opts.device)

    # optimizer, scheduler
    lr_D = 0.0015
    lr_G = 0.0015
    optim_D = torch.optim.Adam(D.parameters(), lr=lr_D, betas=(0.9, 0.999))
    params_G = [{"params": G.g_synthesis.parameters()},
				{"params": G.g_mapping.parameters(), "lr": lr_G * 0.01}]
    optim_G = torch.optim.Adam(params_G, lr=lr_G, betas=(0.9, 0.999))
    scheduler_D = optim.lr_scheduler.ExponentialLR(optim_D, gamma=0.99)
    scheduler_G = optim.lr_scheduler.ExponentialLR(optim_G, gamma=0.99)

    # start training
    fix_z = torch.randn([opts.batch_size, 512]).to(opts.device)
    softplus = torch.nn.Softplus()
    Loss_D_list = [0.0]
    Loss_G_list = [0.0]
    for ep in range(start_epoch, opts.epoch):
        bar = tqdm(loader)
        loss_D_list = []
        loss_G_list = []
        for i, (real_img,) in enumerate(bar):

            real_img = real_img.to(opts.device)
            latents = torch.randn([real_img.size(0), 512]).to(opts.device)

            # Discriminator Network
            real_img = real_img.to(opts.device)
            real_logit = D(real_img)
            fake_img, fake_dlatent = G(latents)
            fake_logit = D(fake_img.detach())

            d_loss = softplus(fake_logit)
            d_loss = d_loss + softplus(-real_logit)

            r1_penalty = D_logistic_r1(real_img.detach(), D)
            d_loss = (d_loss + r1_penalty).mean()

            loss_D_list.append(d_loss.mean().item())

            optim_D.zero_grad()
            d_loss.backward()
            optim_D.step()

            # Generator Network
            G.zero_grad()
            fake_scores_out = D(fake_img)
            _g_loss = softplus(-fake_scores_out)

            g_loss = _g_loss.mean()
            loss_G_list.append(g_loss.mean().item())

            g_loss.backward()
            optim_G.step()

            bar.set_description("Epoch {} [{}, {}] [G]: {} [D]: {}".format(ep, i + 1, len(loader), loss_G_list[-1], loss_D_list[-1]))

        # save result
        Loss_G_list.append(np.mean(loss_G_list))
        Loss_D_list.append(np.mean(loss_D_list))

        with torch.no_grad():
            fake_img = G(fix_z)[0].detach().cpu()
            save_image(fake_img, os.path.join(opts.det, 'images', str(ep) + '.png'), nrow=4, normalize=True)

        # save model
        state = {
            'G': G.state_dict(),
            'D': D.state_dict(),
            'Loss_G': Loss_G_list,
            'Loss_D': Loss_D_list,
            'start_epoch': ep,
        }
        torch.save(state, os.path.join(opts.det, 'models', 'latest.pth'))

        scheduler_D.step()
        scheduler_G.step()

    Loss_D_list = Loss_D_list[1:]
    Loss_G_list = Loss_G_list[1:]
    plotLossCurve(opts, Loss_D_list, Loss_G_list)


if __name__ == '__main__':
    opts = TrainOptions().parse()
    main(opts)
