# main procedure for running training and testing of the proposed segmentation model
# created by Ruiyang Liu
# modified by Han Huang, Xiang Li, and Yuanbiao Wang


import os
import ast
import cv2
import sys
import math
import json
import gzip
import random
import numpy as np
import jittor as jt   
import datetime
import argparse
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from jittor import nn, Module, init, transform  
from jittor.dataset import Dataset

from model import summary
from model import UNet
from model import HRNet
from model import SERT
from loss import CrossEntropyLoss, IoULoss, DiceLoss, FocalLoss
from utils import timewrapper, setupLogger
from utils import SingleDataset, PaintContourDataset, isImageFile
from utils import Evaluator
from advance import STNWrapper, AugDataset, ColorJitter, RandomApply, normalize, aug_for_unet
plt.switch_backend('agg')


modelSet = ['unet', 'hrnet', 'setr', 'aug_unet', 'stn_unet', 'ssl_unet', 'ssl_stn_unet', 'aug_ssl_stn_unet', 'aff_unet', 'aff_ssl_stn_unet']


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='unet', type=str, 
                    choices=modelSet)
parser.add_argument('--mode', type=str, choices=['train', 'test', 'train-test', 'predict', 'debug', 'test_zs_big', 'test_zs_small', 'test_hard'], required=True)
parser.add_argument('--gpu', default=1, type=int, choices=[0, 1])
parser.add_argument('-o', '--optimizer', default='Adam', type=str, choices=['Adam', 'SGD'], dest='optimizer')
parser.add_argument('-e', '--epochs', type=int, default=20,
                    help='Number of epochs', dest='epochs')
parser.add_argument('-b', '--batch-size', type=int, default=8,
                    help='Batch size', dest='batch_size')
parser.add_argument('-l', '--learning-rate', type=float, default=1e-4,
                    help='Learning rate', dest='lr')
parser.add_argument('-p', '--port', type=int,default=10001,
                    help='Visualization port', dest='port')
parser.add_argument('-c', '--class-num', type=int, default=2,
                    help='class number', dest='class_num')
parser.add_argument('-s', '--sequence', default=False, type=ast.literal_eval, choices=[True, False], help="sequence model", dest='seq')

loss_help_msg = """
    Choose from 'ce', 'iou', 'dice', 'focal', 
    or combine with '_' and ratio, like 'ce_0.8_dice_0.2', etc.
    Each loss should be followed with float number as ratio.
"""
parser.add_argument('--loss', type=str, default='ce', help=loss_help_msg)
args = parser.parse_args()

jt.flags.use_cuda = int(args.gpu)

print('======='*10, '\n' , 'args:\n', str(args).replace('Namespace','\t').replace(", ", ",\n\t"), '\n' , '======='*10)

# ========================================================================================================== #

def poly_lr_scheduler(opt, init_lr, iter, epoch, max_iter, max_epoch):
    new_lr = init_lr * (1 - float(epoch * max_iter + iter) / (max_epoch * max_iter)) ** 0.9
    opt.lr = new_lr


criterion_dict = {
    'ce':CrossEntropyLoss(), 
    'iou':IoULoss(), 
    'dice':DiceLoss(), 
    'focal':FocalLoss()
}

criterions = [] # target loss functions designated in the arguments
ratio = [] # ratio for each target function

arg_loss = args.loss.split('_')
if len(arg_loss) == 1:
    assert arg_loss[0] in criterion_dict.keys(), loss_help_msg
    criterions.append(criterion_dict[arg_loss[0]])
    ratio.append(1)
else:
    assert len(arg_loss) % 2 == 0, loss_help_msg
    for i in range(0, len(arg_loss), 2):
        assert arg_loss[i] in criterion_dict.keys(), loss_help_msg
        criterions.append(criterion_dict[arg_loss[i]])
        ratio.append(float(arg_loss[i+1]))


# computes weighted sum of loss functions
def cal_loss(input, target):
    loss = 0
    for i, l in enumerate(criterions):
        loss += ratio[i] * l(input, target)
    return loss


# train function
def train(model, train_loader, optimizer, epoch, init_lr):
    model.train()
    max_iter = len(train_loader)
    
    loss_list = []
    pbar = tqdm(total = max_iter, desc=f"epoch {epoch} train")
    for idx, (imgs, target) in enumerate(train_loader):
        imgs = imgs.float32()
        # if using random affine transformation for augmentation
        # in practise, this augmentation is detrimental to the model, so we do not recommend its usage
        if args.model == 'aff_ssl_stn_unet' or 'aff_unet':
            theta = jt.randn((imgs.shape[0], 2, 3))
            grid = nn.affine_grid(theta, imgs.size())
            imgs = nn.grid_sample(imgs, grid)
            target = target.reshape(imgs.shape[0], 1, 512, 512).float32()
            grid_target = nn.affine_grid(theta, target.size())
            target = nn.grid_sample(target, grid_target).reshape(imgs.shape[0], 512, 512).int()
        pred = model(imgs)
        loss = cal_loss(pred, target)
        optimizer.step(loss)
        loss_list.append(loss.data[0])
        pbar.set_postfix({'loss': loss_list[-1]})
        pbar.update(1)

        # del temporary outputs and loss to save GPU memories
        del pred, loss
    pbar.close()
    return np.mean(loss_list)


# validation function
def val(model, val_loader, epoch, evaluator, best_miou):
    model.eval()
    evaluator.reset()

    n_val = len(val_loader)
    pbar = tqdm(total = n_val, desc=f"epoch {epoch} valid")
    dsc_list = []
    for idx, (imgs, target) in enumerate(val_loader):
        imgs = imgs.float32()
        output = model(imgs)
        pred = output.data
        target = target.data
        pred = np.argmax(pred, axis=1)
        evaluator.add_batch(target, pred)
        dsc = evaluator.calDSC(target, pred)
        dsc_list.append(dsc)
        pbar.update(1)
    pbar.close()
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

    if (mIoU > best_miou):
        best_miou = mIoU
        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')
        if not os.path.exists(f'./checkpoints/{args.model}'):
            os.mkdir(f'./checkpoints/{args.model}')
        model_path = f'./checkpoints/{args.model}/{args.model}-{args.optimizer}-{args.loss}.pkl'
        model.save(model_path)
    print ('Testing result of epoch {}: miou = {} Acc = {} Acc_class = {} FWIoU = {} Best Miou = {} DSC = {}'.format(epoch, mIoU, Acc, Acc_class, FWIoU, best_miou, np.mean(dsc_list)))
    return best_miou, mIoU, np.mean(dsc_list)


# test function
def test(model, test_loader, evaluator):
    model.eval()
    evaluator.reset()

    n_test = len(test_loader)
    dsc_list = []
    recall_list = []
    precision_list = []
    distance_list = []

    pbar = tqdm(total = n_test, desc=f"test")
    for idx, (imgs, target) in enumerate(test_loader):
        output = model(imgs)
        pred = output.data
        target = target.data
        pred = np.argmax(pred, axis=1)
        evaluator.add_batch(target, pred)

        dsc_list.append(evaluator.calDSC(target, pred))
        recall_list.append(evaluator.calRecall(target, pred))
        precision_list.append(evaluator.calPrecision(target, pred))
        distance_list.append(evaluator.calDistance(np.array(target), np.array(pred))[0])
        pbar.update(1)
    pbar.close()

    distance_list = [x for x in distance_list if x]

    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

    result = {
        "mDSC": np.mean(dsc_list),
        'mIoU': mIoU,
        'mFWIoU': FWIoU,
        "mPrecision": np.mean(precision_list),
        "mRecall": np.mean(recall_list),
        "mDistance": np.mean(distance_list),
        "mAcc": Acc,
        "mAcc_class": Acc_class,
    }
    print ('Testing result of {}: miou = {} Acc = {} Acc_class = {} FWIoU = {} mDSC = {} mPrecision = {} mRecall = {} mDistance = {}'.format(
            args.model, mIoU, Acc, Acc_class, FWIoU, np.mean(dsc_list), np.mean(precision_list), np.mean(recall_list), np.mean(distance_list)
        ))

    if not os.path.exists('./result'):
        os.mkdir('./result')
    json.dump(result, open(f"./result/{args.model}-{args.optimizer}-{args.loss}-{args.epochs}-{args.mode}.json","w"), indent=2, ensure_ascii=False)
    return result


# paint contours
def paintContour(model, paint_loader, mask_flag):
    pbar = tqdm(total = len(paint_loader), desc=f"paint coutour")

    # if groundtruth mask is available, draw both the predicted contour and the mask contour
    if mask_flag:
        for idx, (img_file_name, img, img_, mask) in enumerate(paint_loader):
            img_file_name = img_file_name[0]
            img = np.array(img[0])  
            
            pred = model(img_)
            pred = np.argmax(pred, axis=1)
            pred = np.array(pred).astype(np.uint8)  
            contours, hierarchy = cv2.findContours(pred[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)   # findContours shape = (512, 512)
            for contour in contours:
                for point in contour:
                    x, y = point[0]
                    img[y, x, :] = [255, 255, 0]

            mask = np.array(mask).astype(np.uint8)
            contours, hierarchy = cv2.findContours(mask[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)   # findContours shape = (512, 512)
            for contour in contours:
                for point in contour:
                    x, y = point[0]
                    img[y, x, :] = [0, 0, 255]

            if not os.path.exists('./test_performance'):
                os.mkdir('./test_performance')
            if not os.path.exists(f'./test_performance/{args.model}'):
                os.mkdir(f'./test_performance/{args.model}')
            dst = os.path.join('test_performance', args.model, img_file_name.split('/')[-1]).replace('.jpg', '.png')
            cv2.imwrite(dst, img)
            pbar.update()

    # elsewise, simply draw the predicted contour
    else:
        for idx, (img_file_name, img, img_) in enumerate(paint_loader):
            img_file_name = img_file_name[0]
            img = np.array(img[0])   

            pred = model(img_)
            pred = np.argmax(pred, axis=1)
            pred = np.array(pred).astype(np.uint8)
            contours, hierarchy = cv2.findContours(pred[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                for point in contour:
                    x, y = point[0]
                    img[y, x, :] = [255, 255, 0]

            if not os.path.exists('./real_performance'):
                os.mkdir('./real_performance')
            if not os.path.exists(f'./real_performance/{args.model}'):
                os.mkdir(f'./real_performance/{args.model}')
            
            root_path = f'./real_performance/{args.model}'
            sub_files = img_file_name.split('/')
            for sub_file in sub_files:
                if isImageFile(sub_file):
                    break
                root_path = root_path + '/' + sub_file
                if not os.path.exists(root_path):
                    os.mkdir(root_path)
            dst = os.path.join(root_path, img_file_name.replace('.jpg', '.png'))
            cv2.imwrite(dst, img)
            pbar.update()
    pbar.close()


# visualize the contour plotting result
def paintResult(epoch_index_list, epoch_loss_list, epoch_miou_list, epoch_mdsc_list):
    plt.figure(figsize=(20,30), dpi=80, facecolor = "white")
    ax1 = plt.subplot(3,1,1)
    ax1.set_title("Loss")
    plt.plot(epoch_index_list, epoch_loss_list)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    ax2 = plt.subplot(3,1,2)
    ax2.set_title("mIoU")
    plt.plot(epoch_index_list, epoch_miou_list)
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.legend()
    ax3 = plt.subplot(3,1,3)
    ax3.set_title("mDSC")
    plt.plot(epoch_index_list, epoch_mdsc_list)
    plt.xlabel('Epochs')
    plt.ylabel('mDSC')
    plt.legend()
    plt.savefig(f'./result/{args.model}-{args.optimizer}-{args.loss}-{args.epochs}.png')

# ========================================================================================================== #

aug = None
if args.model == 'unet':
    model = UNet(n_channels = 3, n_classes = args.class_num, bilinear = True)
elif args.model == 'ssl_unet':
    model = UNet(n_channels = 3, n_classes = args.class_num, bilinear = True)
    model.load('checkpoints/unet_ssl.pkl')
elif args.model == 'ssl_stn_unet':
    unet = UNet(n_channels = 3, n_classes = args.class_num, bilinear = True)
    unet.load('checkpoints/unet_ssl.pkl')
    model = STNWrapper(unet)
elif args.model == 'hrnet':
    model = HRNet(in_ch=3, out_ch = args.class_num)
elif args.model == 'setr':
    model = SERT(
        patch_size=(32, 32), 
        in_channels=3, 
        out_channels=args.class_num, 
        hidden_size=1024, 
        num_hidden_layers=8, 
        num_attention_heads=16, 
        decode_features=[512, 256, 128, 64]
    )
elif args.model == 'stn_unet':
    unet = UNet(n_channels = 3, n_classes = args.class_num, bilinear = True)
    model = STNWrapper(unet)
elif args.model == 'stn_hrnet':
    hrnet = HRNet(in_ch=3, out_ch = args.class_num)
    model = STNWrapper(hrnet)
elif args.model == 'aug_unet' or args.model == 'aff_unet':
    model = UNet(n_channels = 3, n_classes = args.class_num, bilinear = True)
    aug = aug_for_unet
elif args.model == 'aug_ssl_stn_unet' or args.model == 'aff_ssl_stn_unet':
    unet = UNet(n_channels = 3, n_classes = args.class_num, bilinear = True)
    unet.load('checkpoints/unet_ssl.pkl')
    model = STNWrapper(unet) 
    aug = aug_for_unet
else:
    print("Error: the designated model is not among the available choices")
    exit(0)


# optimizer
batch_size = args.batch_size
lr = args.lr
if args.optimizer == 'SGD':
    optimizer = nn.SGD(model.parameters(), lr, momentum = 0.9, weight_decay = 1e-4)
else:
    optimizer = nn.Adam(model.parameters(), lr, betas = (0.9,0.999))
    

# data and logging
epochs = args.epochs
best_miou = 0.0
best_mdsc = 0.0
epoch_index_list = list(range(epochs))
epoch_loss_list = []
epoch_miou_list = []
epoch_mdsc_list = []
evaluator = Evaluator(num_class = args.class_num)

train_XH = '../data/data-XH/train_label.json'
val_XH   = '../data/data-XH/val_label.json'
test_XH  = '../data/data-XH/test_label.json'
img_XH   = '../data/data-XH/data'
mask_XH  = '../data/data-XH/label'
test_ZS_Big = '../data/data-ZS/Big/label.json'
img_ZS_Big  = '../data/data-ZS/Big/data'
mask_ZS_Big = '../data/data-ZS/Big/label'
test_ZS_Small = '../data/data-ZS/Small/label.json'
img_ZS_Small  = '../data/data-ZS/Small/data'
mask_ZS_Small = '../data/data-ZS/Small/label'
test_hard = '../data/data-hard/label.json'
img_hard = '../data/data-hard/data'
mask_hard = '../data/data-hard/label'


# main loop
if args.mode == 'train-test':
    train_loader = SingleDataset(json_dir = train_XH, img_dir = img_XH, mask_dir = mask_XH, batch_size = args.batch_size, shuffle = True, train = True, aug=aug)
    val_loader = SingleDataset(json_dir = val_XH, img_dir = img_XH, mask_dir = mask_XH, batch_size = 1, shuffle = False, train = False)
    test_loader = SingleDataset(json_dir = test_XH, img_dir = img_XH, mask_dir = mask_XH, batch_size = 1, shuffle = False, train = False)
    paint_loader = PaintContourDataset(json_dir = test_XH, img_dir = img_XH, mask_dir = mask_XH, batch_size = 1, shuffle = False)
    for epoch in range(epochs):
        epoch_loss = train(model, train_loader, optimizer, epoch, lr)
        best_miou, mIoU, mdsc = val(model, val_loader, epoch, evaluator, best_miou)
        epoch_loss_list.append(epoch_loss)
        epoch_miou_list.append(mIoU)
        epoch_mdsc_list.append(mdsc)
    paintResult(epoch_index_list, epoch_loss_list, epoch_miou_list, epoch_mdsc_list)
    result = test(model, test_loader, evaluator)
    paintContour(model, paint_loader, mask_flag = True)

elif args.mode == 'train':
    train_loader = SingleDataset(json_dir = train_XH, img_dir = img_XH, mask_dir = mask_XH, batch_size = args.batch_size, shuffle = True, train = True, aug=aug)
    val_loader = SingleDataset(json_dir = val_XH, img_dir = img_XH, mask_dir = mask_XH, batch_size = 1, shuffle = False, train = False)
    for epoch in range(epochs):
        epoch_loss = train(model, train_loader, optimizer, epoch, lr)
        best_miou, mIoU, mdsc = val(model, val_loader, epoch, evaluator, best_miou)
        epoch_loss_list.append(epoch_loss)
        epoch_miou_list.append(mIoU)
        epoch_mdsc_list.append(mdsc)
    paintResult(epoch_index_list, epoch_loss_list, epoch_miou_list, epoch_mdsc_list)

elif args.mode == 'test':
    model.load_parameters(jt.load(f"./checkpoints/{args.model}/{args.model}-{args.optimizer}-{args.loss}.pkl"))
    test_loader = SingleDataset(json_dir = test_XH, img_dir = img_XH, mask_dir = mask_XH, batch_size = 1, shuffle = False, train = False)
    paint_loader = PaintContourDataset(json_dir = test_XH, img_dir = img_XH, mask_dir = mask_XH, batch_size = 1, shuffle = False)
    result = test(model, test_loader, evaluator)
    paintContour(model, paint_loader, mask_flag = True)
    
# tests on ZhongShan dataset(both small images and large images)
elif args.mode == 'test_zs_big':
    model.load_parameters(jt.load(f"./checkpoints/{args.model}/{args.model}-{args.optimizer}-{args.loss}.pkl"))
    test_loader = SingleDataset(json_dir = test_ZS_Big, img_dir = img_ZS_Big, mask_dir = mask_ZS_Big, batch_size = 1, shuffle = False, train = False)
    paint_loader = PaintContourDataset(json_dir = test_ZS_Big, img_dir = img_ZS_Big, mask_dir = mask_ZS_Big, batch_size = 1, shuffle = False)
    result = test(model, test_loader, evaluator)
    paintContour(model, paint_loader, mask_flag = True)
    
elif args.mode == 'test_zs_small':
    model.load_parameters(jt.load(f"./checkpoints/{args.model}/{args.model}-{args.optimizer}-{args.loss}.pkl"))
    test_loader = SingleDataset(json_dir = test_ZS_Small, img_dir = img_ZS_Small, mask_dir = mask_ZS_Small, batch_size = 1, shuffle = False, train = False)
    paint_loader = PaintContourDataset(json_dir = test_ZS_Small, img_dir = img_ZS_Small, mask_dir = mask_ZS_Small, batch_size = 1, shuffle = False)
    result = test(model, test_loader, evaluator)
    paintContour(model, paint_loader, mask_flag = True)
    
elif args.mode == 'test_hard':
    model.load_parameters(jt.load(f"./checkpoints/{args.model}/{args.model}-{args.optimizer}-{args.loss}.pkl"))
    test_loader = SingleDataset(json_dir = test_hard, img_dir = img_hard, mask_dir = mask_hard, batch_size = 1, shuffle = False, train = False)
    paint_loader = PaintContourDataset(json_dir = test_hard, img_dir = img_hard, mask_dir = mask_hard, batch_size = 1, shuffle = False)
    result = test(model, test_loader, evaluator)
    paintContour(model, paint_loader, mask_flag = True)

elif args.mode == 'predict':
    model.load_parameters(jt.load(f"./checkpoints/{args.model}/{args.model}-{args.optimizer}-{args.loss}.pkl"))
    paint_loader = PaintContourDataset(json_dir = test_XH, img_dir = img_XH, mask_dir = None, batch_size = 1, shuffle = False)
    paintContour(model, paint_loader, mask_flag = False)

else: # args.mode == 'debug'
    x = jt.ones([2, 3, 512, 512])
    y = model(x)
    print(y.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    summary(model, input_size=(3, 512, 512), device='cuda')
