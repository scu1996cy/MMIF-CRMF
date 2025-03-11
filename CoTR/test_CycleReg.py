from __future__ import print_function
import argparse
import os

import torch
import torchvision.transforms as transforms
from utils import is_image_file, load_img, save_img

# Testing settings
parser = argparse.ArgumentParser(description='cyclegan-pytorch-implementation')
parser.add_argument('--name', type=str, default='CycleReg', help='facades')
parser.add_argument('--dataset', type=str, default='CoTR', help='pix2pix')
parser.add_argument('--nepochs', type=int, default=200, help='saved model of which epochs')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--mode', type=str, default='test', help='test/train')
opt = parser.parse_args()
print(opt)

GPU_iden = 0
GPU_num = torch.cuda.device_count()
print('Number of GPU: ' + str(GPU_num))
for GPU_idx in range(GPU_num):
    GPU_name = torch.cuda.get_device_name(GPU_idx)
    print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
torch.cuda.set_device(GPU_iden)
GPU_avai = torch.cuda.is_available()
print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
print('If the GPU is available? ' + str(GPU_avai))

g_b2a_model_path = "checkpoint/{}/netG_b2a_model_epoch_{}.pth".format(opt.name, opt.nepochs)
r_a2b_model_path = "checkpoint/{}/netR_a2b_model_epoch_{}.pth".format(opt.name, opt.nepochs)

net_g_b2a = torch.load(g_b2a_model_path).cuda()
net_r_a2b = torch.load(r_a2b_model_path).cuda()

real_a_dir = "dataset/{}/{}/a/".format(opt.dataset, opt.mode)
real_b_dir = "dataset/{}/{}/b/".format(opt.dataset, opt.mode)

image_filenames = [x for x in os.listdir(real_a_dir) if is_image_file(x)]

a_transform_list = [transforms.ToTensor()]
a_transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
a_transform = transforms.Compose(a_transform_list)

b_transform_list = [transforms.Grayscale(1)]
b_transform_list += [transforms.ToTensor()]
b_transform_list += [transforms.Normalize((0.5,), (0.5,))]
b_transform = transforms.Compose(b_transform_list)

for image_name in image_filenames:
    real_a = load_img(real_a_dir + image_name)
    real_b = load_img(real_b_dir + image_name)
    real_a = a_transform(real_a)
    real_b = b_transform(real_b)
    real_a = real_a.unsqueeze(0).cuda()
    real_b = real_b.unsqueeze(0).cuda()

    fake_a = net_g_b2a(real_b)
    warped_real_a, flow_a2b = net_r_a2b(real_a, fake_a)

    fake_a_img = fake_a.detach().squeeze(0).cpu()
    out_img = warped_real_a.detach().squeeze(0).cpu()

    if not os.path.exists(os.path.join("result", opt.name, opt.mode)):
        os.makedirs(os.path.join("result", opt.name, opt.mode))
    save_img(out_img, "result/{}/{}/{}".format(opt.name, opt.mode, image_name))