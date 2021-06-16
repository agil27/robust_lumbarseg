# procedure for the constrastive learning pretraining of the segmentation model
# created by Yuanbiao Wang
# please export the environment variable log_silent=1 before running this script


import jittor as jt
import jittor.nn as nn
import numpy as np
from model import UNet
from advance import CoLearner, AugDataset, TwoCropsTransform, augmentation
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')


if __name__ == '__main__':
    jt.flags.use_cuda = int(1)
    unet = UNet()
    train_XH = '../data/data-XH/train_label.json'
    val_XH   = '../data/data-XH/val_label.json'
    test_XH  = '../data/data-XH/test_label.json'
    img_XH   = '../data/data-XH/data'
    mask_XH  = '../data/data-XH/label'

    train_loader = AugDataset(
        json_dir=train_XH, 
        img_dir=img_XH, 
        mask_dir=mask_XH, 
        batch_size=8, 
        shuffle=True, 
        aug=TwoCropsTransform(augmentation)
    )

    learner = CoLearner(
        model=unet,
        layer='down4',
        loader=train_loader,
        embedding_channel=512,
        project_dim=128
    )

    loss_min = 1e4
    losses = []
    with open('ssl.txt', 'w') as f:
      for epoch in range(25):
          loss = learner.train()
          print('epoch[%02d] loss:[%.6f\n]' % (epoch + 1, loss))
          f.write('epoch[%02d] loss:[%.6f\n]' % (epoch + 1, loss))
          if loss < loss_min:
            unet.save('checkpoints/unet_ssl.pkl')
          losses.append(loss)
    np.savetxt('ssl_loss.txt', loss)
    plt.plot(losses)
    plt.savefig('ssl_losses.png')