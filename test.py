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

from matplotlib import pyplot as plt

from copy import deepcopy

DATA_DIR = '/n/liyz/data/deep-steganography-dataset/'

'''
10-31
1- adjust the image size to be the same as cover
2- modify the logic in test
'''

def viridis_cmap(gray: np.ndarray):
    """
    Visualize a single-channel image using matplotlib's viridis color map
    yellow is high value, blue is low
    :param gray: np.ndarray, (H, W) or (H, W, 1) unscaled
    :return: (H, W, 3) float32 in [0, 1]
    """
    colored = plt.cm.viridis(plt.Normalize()(gray.squeeze()))[..., :-1]
    # pdb.set_trace()
    return colored.astype(np.float32)

def unscaled_viridis_cmap(gray: np.ndarray, scale=1):
    """
    Visualize a single-channel image using matplotlib's viridis color map
    yellow is high value, blue is low
    :param gray: np.ndarray, (H, W) or (H, W, 1) unscaled
    :return: (H, W, 3) float32 in [0, 1]
    """
    # pdb.set_trace()
    colored = plt.cm.viridis((gray.squeeze())*scale )[..., :-1]
    # pdb.set_trace()
    return colored.astype(np.float32)



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
            cover_img = torch.FloatTensor( torch.from_numpy( cover_img).permute(2,0,1).unsqueeze(0)*1.0/255).cuda() # 1,3,756,1008

            secret_img = imageio.imread('secret_images2/white-100-full.png')
            # secret_img = imageio.imread('secret_images/logo.jpg')
            secret_img = torch.FloatTensor( torch.from_numpy( secret_img).permute(2,0,1).unsqueeze(0)*1.0/255).cuda()

            # secret_img = secret_img.repeat(len(cover_img),1,1,1)

            # concat cover and original secret to get the concat_img with 6 channels
            # cover_img = F.interpolate(cover_img,(768,1024)) #1,3, 768,1024
            # secret_img = F.interpolate(secret_img,(768,1024))

            # cover_img = F.interpolate(cover_img,(H//128*128,W//128*128)) #1,3, 768,1024
            secret_img = F.interpolate(secret_img,(H_WM, W_WM)) 

            concat_img = torch.cat([cover_img[:,:,:H_WM, :W_WM], secret_img], dim=1) #1,6,640,896

            # if opt.cuda:
            #     cover_img = cover_img.cuda()
            #     secret_img = secret_img.cuda()
            #     concat_img = concat_img.cuda()

            concat_imgv = Variable(concat_img, volatile=True)  # concat_img as input of Hiding net
            # cover_imgv = Variable(cover_img, volatile=True)  # cover_imgv as label of Hiding net

            container_img = Hnet(concat_imgv)  # take concat_img as input of H-net and get the container_img
            # errH = criterion(container_img, cover_imgv)  # H-net reconstructed error
            

            # container_img = imageio.imread('results/re_cover_1.png')
            # container_img = torch.FloatTensor( torch.from_numpy(container_img).permute(2,1,0).unsqueeze(0)*1.0/255).cuda()


            # _container_img = imageio.imread('results/re_cover_0.png')
            # _container_img = torch.FloatTensor( torch.from_numpy( _container_img).permute(2,1,0).unsqueeze(0)*1.0/255).cuda()

            # _container_img = F.interpolate(_container_img,(1024,768))

            # container_img = Variable(container_img, volatile=True)  # 1,3,768,1024
            rev_secret_img = Rnet(container_img)  # containerImg as input of R-net and get "rev_secret_img"  # 1,3,768,1024
            # secret_imgv = Variable(secret_img, volatile=True)  # secret_imgv as label of R-net
            # errR = criterion(rev_secret_img, secret_imgv)  # R-net reconstructed error

            # re_covers = F.interpolate(container_img,(756,1008)) # 1,3,756,1008
            # re_covers = container_img
            re_covers = deepcopy(cover_img)
            re_covers[:,:,:H_WM, :W_WM] = container_img

            re_secrets = F.interpolate(rev_secret_img,(128,128)) # 1,3,128,128

            imageio.imwrite( f'temp_cover_1.png', re_covers[0].permute(1,2,0).cpu().numpy() )
            imageio.imwrite( f'temp_secret_1.png', re_secrets[0].permute(1,2,0).cpu().numpy() )


            residual_map1 = unscaled_viridis_cmap( torch.abs(cover_img-re_covers).squeeze().permute(1,2,0).cpu().numpy().mean(-1), scale=25)
            residual_map2 = viridis_cmap( torch.abs(cover_img-re_covers).squeeze().permute(1,2,0).cpu().numpy().mean(-1))


            imageio.imwrite('res1.png', residual_map1)
            imageio.imwrite('res2.png', residual_map2)
#
            


            # container_img = imageio.imread('temp_cover_0.png')  #768,1024,3
            # container_img = torch.FloatTensor( torch.from_numpy( container_img).permute(2,0,1).unsqueeze(0)*1.0/255).cuda()

            # # container_img = F.interpolate(container_img,(768,1024)) #1,3, 768,1024

            # rev_secret_img = Rnet(container_img[:,:,:H_WM, :W_WM])  # containerImg as input of R-net and get "rev_secret_img"
            # # secret_imgv = Variable(secret_img, volatile=True)  # secret_imgv as label of R-net
            # # errR = criterion(rev_secret_img, secret_imgv)  # R-net reconstructed error

            # container_img = F.interpolate(container_img,(H,W))
            # re_secrets = F.interpolate(rev_secret_img,(128,128))

            # imageio.imwrite( f'temp_cover_1.png', container_img[0].permute(1,2,0).cpu().numpy() )
            # imageio.imwrite( f'temp_secret_1.png', re_secrets[0].permute(1,2,0).cpu().numpy() )


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


    print (cnt)




if __name__ == '__main__':
    main()
