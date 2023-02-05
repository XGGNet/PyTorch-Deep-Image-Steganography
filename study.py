# encoding: utf-8
"""
@author: yongzhi li
@contact: yongzhili@vip.qq.com

@version: 1.0
@file: main.py
@time: 2018/3/20

"""

import argparse
import os
import shutil
import socket
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

import torch.nn.functional as F

from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import utils.transformed as transforms
from data.ImageFolderDataset import MyImageFolder
from models.HidingUNet import UnetGenerator
from models.RevealNet import RevealNet

import imageio
import cv2
import numpy as np

from copy import deepcopy

import pdb

# import sys

# sys.setrecursionlimit(1000000) #例如这里设置为一百万


import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

DATA_DIR = '/n/liyz/data/deep-steganography-dataset/'

'''
10-31
1- adjust the image size to be the same as cover
2- modify the logic in test
'''

def add_noise(image, degree=0.1):
    # 1. Add noises to the image
    noisy1 = image + image.std() * np.random.random(image.shape)

    alot  = image.max() * np.random.random(image.shape)
    noisy2 = image + 0.1*alot

    return noisy2

def gaussian_blur(src, kernel_size, sigma):
    image = src
    if sigma==0:
        sigma=0.01
    dst = cv2.GaussianBlur(image, (kernel_size,kernel_size), sigma)
    return dst


def jpeg(img, degree=0.95):

    # pdb.set_trace()

    # try:

    # assert len(img_tensor.shape) == 3
    # assert img_tensor.shape[-1] == 3

    # print(img.min(), img.max())

    img_tensor = torch.from_numpy(img)

    tensor_max = img_tensor.max()
    tensor_min = img_tensor.min()

    img_tensor = (img_tensor - tensor_min) / (tensor_max - tensor_min)

    np_img = (img_tensor.detach().cpu().numpy()*255).astype(np.uint8)
    

    retain_degree = int((1-degree)*100)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), retain_degree] 

    # pdb.set_trace()

    result, encode_img = cv2.imencode('.jpg', np_img, encode_param)
    
    decode_img = cv2.imdecode(encode_img, 1)

    decode_img = (decode_img*1.0/255).astype(np.float32)

    decode_tensor = torch.from_numpy(decode_img)

    # 

    decode_tensor = decode_tensor*(tensor_max.cpu() - tensor_min.cpu()) + tensor_min.cpu()

    
    # pdb.set_trace()        
    # except:
    #     print('####', img_tensor.shape, img_tensor.max(), img_tensor.min(), retain_degree)

    # print(decode_tensor.min(), decode_tensor.max())
    # pdb.set_trace()
    return decode_tensor.cpu().numpy()


def ssim_function(img1, img2, window_size = 11, size_average = True, format='NCHW'):
    if format == 'HWC':
        img1 = img1.permute([2, 0, 1])[None, ...]
        img2 = img2.permute([2, 0, 1])[None, ...]
    elif format == 'NHWC':
        img1 = img1.permute([0, 3, 1, 2])
        img2 = img2.permute([0, 3, 1, 2])

    return ssim(img1, img2, window_size, size_average)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="train",
                    help='train | val | test')
parser.add_argument('--workers', type=int, default=8,
                    help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=4,
                    help='input batch size')
parser.add_argument('--imageSize', type=int, default=256,
                    help='the number of frames')

parser.add_argument('--imageH', type=int, default=1024, # 1008
                    help='the number of frames')
parser.add_argument('--imageW', type=int, default=768, # 756
                    help='the number of frames')

parser.add_argument('--niter', type=int, default=100,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate, default=0.001')
parser.add_argument('--decay_round', type=int, default=10,
                    help='learning rate decay 0.5 each decay_round')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--Hnet', default='',
                    help="path to Hidingnet (to continue training)")
parser.add_argument('--Rnet', default='',
                    help="path to Revealnet (to continue training)")
parser.add_argument('--trainpics', default='./training/',
                    help='folder to output training images')
parser.add_argument('--validationpics', default='./training/',
                    help='folder to output validation images')
parser.add_argument('--testPics', default='./training/',
                    help='folder to output test images')
parser.add_argument('--outckpts', default='./training/',
                    help='folder to output checkpoints')
parser.add_argument('--outlogs', default='./training/',
                    help='folder to output images')
parser.add_argument('--outcodes', default='./training/',
                    help='folder to save the experiment codes')
parser.add_argument('--beta', type=float, default=0.75,
                    help='hyper parameter of beta')
parser.add_argument('--remark', default='', help='comment')
parser.add_argument('--test', default='cover_images/flower/images_4/', help='test mode, you need give the test pics dirs in this param')

parser.add_argument('--hostname', default=socket.gethostname(), help='the  host name of the running server')
parser.add_argument('--debug', type=bool, default=False, help='debug mode do not create folders')
parser.add_argument('--logFrequency', type=int, default=10, help='the frequency of print the log on the console')
parser.add_argument('--resultPicFrequency', type=int, default=100, help='the frequency of save the resultPic')


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def main():
    ############### define global parameters ###############
    global opt, optimizerH, optimizerR, writer, logPath, schedulerH, schedulerR, val_loader, smallestLoss

    #################  output configuration   ###############
    opt = parser.parse_args()

    opt.Hnet = "./checkPoint/netH_epoch_73,sumloss=0.000447,Hloss=0.000258.pth"
    opt.Rnet = "./checkPoint/netR_epoch_73,sumloss=0.000447,Rloss=0.000252.pth"
    testdir = opt.test

    '''
    cover_folder
    '''
    # test_dataset = MyImageFolder(
    #     testdir,
    #     transforms.Compose([
    #         transforms.Resize([opt.imageW, opt.imageH]), # 有resize
    #         transforms.ToTensor(),
    #     ]))
    
    # secret_dir = 'secret_images'
    # secret_dataset = MyImageFolder(
    #     secret_dir,
    #     transforms.Compose([
    #         transforms.Resize([opt.imageW, opt.imageH]), # 有resize
    #         transforms.ToTensor(),
    #     ]))
    # assert len( secret_dataset) == 1

    # assert test_dataset

    Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=7, output_function=nn.Sigmoid)
    Hnet.cuda()
    Hnet.apply(weights_init)
    # whether to load pre-trained model
    if opt.Hnet != "":
        Hnet.load_state_dict(torch.load(opt.Hnet))
    if opt.ngpu > 1:
        Hnet = torch.nn.DataParallel(Hnet).cuda()


    Rnet = RevealNet(output_function=nn.Sigmoid)
    Rnet.cuda()
    Rnet.apply(weights_init)
    if opt.Rnet != '':
        Rnet.load_state_dict(torch.load(opt.Rnet))
    if opt.ngpu > 1:
        Rnet = torch.nn.DataParallel(Rnet).cuda()


    # MSE loss
    criterion = nn.MSELoss().cuda()


    # test_loader = DataLoader(test_dataset, batch_size=opt.batchSize,
    #                             shuffle=False, num_workers=int(opt.workers))
    # secret_loader = DataLoader(secret_dataset, batch_size=opt.batchSize,
    #                             shuffle=False, num_workers=int(opt.workers))                     
    
    test(None, None, 0, Hnet=Hnet, Rnet=Rnet, criterion=criterion)
    print("##################   test is completed, the result pic is saved in the ./training/yourcompuer+time/testPics/   ######################")


def test(test_loader, secret_loader, epoch, Hnet, Rnet, criterion):
    print(
        "#################################################### test begin ########################################################")
    start_time = time.time()
    Hnet.eval()
    Rnet.eval()

    cnt = 0
    with torch.no_grad():
        # for i, data in enumerate(test_loader, 0):
        if True:

            Hnet.zero_grad()
            Rnet.zero_grad()

            '''
            必须128的倍数
            '''

            # all_pics = data  # allpics contains cover images and secret images
            # this_batch_size = int(all_pics.size()[0] / 2)  # get true batch size of this step 

            # this_batch_size = int(all_pics.size()[0])
            # secret_img = [data for data in secret_loader][0]

            # first half of images will become cover images, the rest are treated as secret images
            # cover_img = all_pics[0:this_batch_size, :, :, :]  # batchSize,3,256,256
            # secret_img = all_pics[this_batch_size:this_batch_size * 2, :, :, :]
            
            '''
            原始cover img
            '''
            cover_img = imageio.imread('cover_images/flower/images_4/image000.png') #756,1008,3   H,W,C
            H, W = cover_img.shape[0:2]
            H_WM,W_WM = H//128*128,W//128*128
            cover_img = torch.FloatTensor( torch.from_numpy( cover_img).permute(2,0,1).unsqueeze(0)*1.0/255).cuda().float() # 1,3,756,1008

            # secret_img = imageio.imread('secret_images2/white-100-full.png')
            secret_img = imageio.imread('secret_images/logo.jpg')
            secret_img = torch.FloatTensor( torch.from_numpy( secret_img).permute(2,0,1).unsqueeze(0)*1.0/255).cuda().float()

            # pdb.set_trace()

            # secret_img = secret_img.repeat(len(cover_img),1,1,1)

            # concat cover and original secret to get the concat_img with 6 channels
            cover_img = F.interpolate(cover_img,(768,1024)) #1,3, 768,1024
            secret_img = F.interpolate(secret_img,(768,1024))

            # cover_img = F.interpolate(cover_img,(H//128*128,W//128*128)) #1,3, 768,1024
            # secret_img = F.interpolate(secret_img,(H_WM, W_WM)) 

            concat_img = torch.cat([cover_img, secret_img], dim=1) #1,6,640,896

            container_img = Hnet(concat_img)  # take concat_img as input of H-net and get the container_img

            rev_secret_img = Rnet(container_img)  # containerImg as input of R-net and get "rev_secret_img"  # 1,3,

            re_secrets = F.interpolate(rev_secret_img,(128,128)) # 1,3,128,128

            imageio.imwrite( f'study/pure_cover.png', container_img[0].permute(1,2,0).cpu().numpy() )
            imageio.imwrite( f'study/pure_secret.png', re_secrets[0].permute(1,2,0).cpu().numpy() )

            pure_container_img = container_img[0].detach().clone().permute(1,2,0).cpu().numpy()

            # IMG_SSIM = {'noise':[], 'blur':[], 'jpeg':[]}
            # WM_SSIM = {'noise':[], 'blur':[], 'jpeg':[]}


            for ks in [15]:

                IMG_SSIM = {}
                WM_SSIM = {}

                for degree in np.arange(0,5.05,0.25):

                    print(degree)

                    # noise_container_img = add_noise(pure_container_img, degree=degree)

                    blur_container_img = gaussian_blur(pure_container_img, kernel_size=ks, sigma=degree)

                    # JPEG_container_img = jpeg(pure_container_img, degree=degree)

                    


                    # container_img = imageio.imread('temp_cover_0.png')  #768,1024,3
                    # noise_container_img = torch.from_numpy( noise_container_img).permute(2,0,1).unsqueeze(0).cuda()
                    # noise_container_img = noise_container_img.type(torch.FloatTensor).cuda()

                    # noise_rev_secret_img = Rnet(noise_container_img)  # containerImg as input of R-net and get "rev_secret_img"
                    # # secret_imgv = Variable(secret_img, volatile=True)  # secret_imgv as label of R-net
                    # # errR = criterion(rev_secret_img, secret_imgv)  # R-net reconstructed error

                    # imageio.imwrite( f'study2/noise-{int(degree*100)}_cover.png', noise_container_img[0].permute(1,2,0).cpu().numpy() )
                    # imageio.imwrite( f'study2/noise-{int(degree*100)}_secret.png', noise_rev_secret_img[0].permute(1,2,0).cpu().numpy() )


                    blur_container_img = torch.from_numpy( blur_container_img).permute(2,0,1).unsqueeze(0).cuda()
                    blur_container_img = blur_container_img.type(torch.FloatTensor).cuda()

                    blur_rev_secret_img = Rnet(blur_container_img)  # containerImg as input of R-net and get "rev_secret_img"

                    blur_rev_secret_img = F.interpolate(blur_rev_secret_img,(128,128))

                    # secret_imgv = Variable(secret_img, volatile=True)  # secret_imgv as label of R-net
                    # errR = criterion(rev_secret_img, secret_imgv)  # R-net reconstructed error

                    save_dir = f'study-black_blur-k{ks}/'
                    os.makedirs(save_dir, exist_ok=True)

                    imageio.imwrite( os.path.join(save_dir, f'blur-{int(degree*100)}_cover.png'), blur_container_img[0].permute(1,2,0).cpu().numpy() )
                    imageio.imwrite( os.path.join(save_dir, f'blur-{int(degree*100)}_secret.png'), blur_rev_secret_img[0].permute(1,2,0).cpu().numpy() )

                    # pdb.set_trace()

                    IMG_SSIM[str(degree)] = ssim_function( blur_container_img.squeeze().permute(1,2,0), torch.from_numpy(pure_container_img).float().cuda(),format='HWC')

                    WM_SSIM[str(degree)] = ssim_function( blur_rev_secret_img.detach().cpu().squeeze().permute(1,2,0), re_secrets.detach().cpu().squeeze().permute(1,2,0),format='HWC' )

                    # pdb.set_trace()
                print(f'########ks={ks}')
                print('########### IMG_SSIM')
                for key, value in IMG_SSIM.items():
                    print(value.item())
                print('\n########### WM_SSIM')
                for key, value in WM_SSIM.items():
                    print(value.item())

                pdb.set_trace()

                # JPEG_container_img = torch.from_numpy( JPEG_container_img).permute(2,0,1).unsqueeze(0).cuda()
                # JPEG_container_img = JPEG_container_img.type(torch.FloatTensor).cuda()

                # JPEG_rev_secret_img = Rnet(JPEG_container_img)  # containerImg as input of R-net and get "rev_secret_img"
                # # secret_imgv = Variable(secret_img, volatile=True)  # secret_imgv as label of R-net
                # # errR = criterion(rev_secret_img, secret_imgv)  # R-net reconstructed error

                # imageio.imwrite( f'study2/JPEG-{int(degree*100)}_cover.png', JPEG_container_img[0].permute(1,2,0).cpu().numpy() )
                # imageio.imwrite( f'study2/JPEG-{int(degree*100)}_secret.png', JPEG_rev_secret_img[0].permute(1,2,0).cpu().numpy() )


                # container_img = imageio.imread('cover_images/flower/images_4/image000.png') #756,1008,3
                # container_img = torch.FloatTensor( torch.from_numpy( container_img).permute(2,1,0).unsqueeze(0)*1.0/255).cuda()

                # rev_secret_img = Rnet(container_img)  # containerImg as input of R-net and get "rev_secret_img"
                # # secret_imgv = Variable(secret_img, volatile=True)  # secret_imgv as label of R-net
                # # errR = criterion(rev_secret_img, secret_imgv)  # R-net reconstructed error

                # re_covers = F.interpolate(container_img,(1008,756))
                # re_secrets = F.interpolate(rev_secret_img,(128,128))

                # imageio.imwrite( f'temp_cover_2.png', container_img[0].permute(2,1,0).cpu().numpy() )
                # imageio.imwrite( f'temp_secret_2.png', re_secrets[0].permute(2,1,0).cpu().numpy() )
        

                # for jk in range( len(re_covers) ):
                #     imageio.imwrite( f'results/re_cover_{cnt}.png', re_covers[jk].permute(1,2,0).cpu().numpy() )
                #     imageio.imwrite( f'results/re_secret_{cnt}.png', re_secrets[jk].permute(1,2,0).cpu().numpy() )
                #     cnt += 1


    # print (cnt)




if __name__ == '__main__':
    main()
