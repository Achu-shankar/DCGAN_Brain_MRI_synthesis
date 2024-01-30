import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from config import *
from DCGAN import Generator

pretrained_model_folder = 'pretrained_models'
pretrained_model_path = os.path.join(pretrained_model_folder,'DCGAN_brain_mri_model_weights.pth')
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Create the generator
netG = Generator(ngpu).to(device)
# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

checkpoint = torch.load(pretrained_model_path)
netG.load_state_dict(checkpoint['modelG_state_dict'])

fixed_noise = torch.randn(128, nz, 1, 1, device=device)
img_list = []
with torch.no_grad():
    fake = netG(fixed_noise).detach().cpu()
img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

# Plot the fake images from the last epoch
plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()