# This is part of the tutorial materials in the UCL Module MPHY0041: Machine Learning in Medical Imaging
import os
import torch
import numpy as np
from tqdm import tqdm
import random
import copy
from utils.average_meter import AverageMeter
from utils.saver import Saver
import click
import seg_metrics.seg_metrics as sg
from models import UNet



os.environ["CUDA_VISIBLE_DEVICES"]="0"
use_cuda = torch.cuda.is_available()
folder_name = './data/datasets-promise12'
RESULT_PATH = './result'
## network class


## loss function
def loss_dice(y_pred, y_true, eps=1e-6):
    '''
    y_pred, y_true -> [N, C=1, D, H, W]
    '''
    numerator = torch.sum(y_true*y_pred, dim=(2,3,4)) * 2
    denominator = torch.sum(y_true, dim=(2,3,4)) + torch.sum(y_pred, dim=(2,3,4)) + eps
    return torch.mean(1. - (numerator / denominator))

# soft loss function
def soft_dice_loss(y_pred, y_true, eps = 1e-6):
    '''
        y_pred, y_true -> [N, C=1, D, H, W]
    '''
    numerator =  torch.sum(y_true * y_pred, dim=(2, 3, 4)) * 2 + eps
    denominator = torch.sum(y_true**2, dim=(2, 3, 4)) + torch.sum(y_pred**2, dim=(2, 3, 4)) + eps
    return torch.mean(1. - (numerator / denominator))

# dice_score
def compute_dice_score(y_pred, y_true, eps=1e-6):
    '''
    y_pred, y_true -> [N, C=1, D, H, W]
    '''
    numerator = torch.sum(y_true*y_pred, dim=(2,3,4)) * 2
    denominator = torch.sum(y_true, dim=(2,3,4)) + torch.sum(y_pred, dim=(2,3,4)) + eps
    return torch.mean(numerator / denominator)

# Intersection over Union (Jaccard Index)
def iou_coef(y_true, y_pred, smooth=1):
  intersection = torch.sum(torch.abs(y_true * y_pred), dim=[2,3,4])
  union = torch.sum(y_true,[2,3,4])+torch.sum(y_pred,[2,3,4])-intersection
  iou = torch.mean((intersection + smooth) / (union + smooth))
  return iou

## data loader
class NPyDataset(torch.utils.data.Dataset):
    def __init__(self, folder_name, is_train=True):
        self.folder_name = folder_name
        self.is_train = is_train

    def __len__(self):
        return (50 if self.is_train else 30)

    def __getitem__(self, idx):
        if self.is_train:
            image = self._load_npy("image_train%02d.npy" % idx)
            label = self._load_npy("label_train%02d.npy" % idx)
            return image, label, idx
        else:
            return self._load_npy("image_test%02d.npy" % idx), idx

    def _load_npy(self, filename):
        filename = os.path.join(self.folder_name, filename)
        return torch.unsqueeze(torch.tensor(np.float32(np.load(filename))),dim=0) # full size volume
        #return torch.unsqueeze(torch.tensor(np.float32(np.load(filename)[::2,::2,::2])),dim=0) #sampled volume


loss_dict = {
    'dice': loss_dice,
    'soft_dice': soft_dice_loss
}

@click.command()
@click.option('--loss_type', default = 'dice')
@click.option('--exp_name', default = 'test')
def main(loss_type, exp_name):

    loss_fn = loss_dict[loss_type]
    saver = Saver('runs', exp_name)
    ## training
    model = UNet(1,1)  # input 1-channel 3d volume and output 1-channel segmentation (a probability map)
    if use_cuda:
        model.cuda()

    # train/val split
    train_set = NPyDataset(folder_name)
    train_idxs = [i for i in range(50)]
    random.Random(99).shuffle(train_idxs)
    val_set = torch.utils.data.Subset(train_set,train_idxs[:10])
    new_train_set = torch.utils.data.Subset(train_set,train_idxs[10:])

    train_loader = torch.utils.data.DataLoader(
        new_train_set,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        persistent_workers = True,
        pin_memory= True)

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=5,
        shuffle=False,
        num_workers=1,
        persistent_workers = True,
        pin_memory= True)

    val_loader_final = torch.utils.data.DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=1)


    # optimisation loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loaders = {
        'train': train_loader,
        'val': val_loader
    }

    # Best Model Selection (min val loss)
    min_val_acc = float('inf')
    min_val_epoch = None
    best_model = None

    

    for epoch in tqdm(range(1000)):
        for split, loader in loaders.items():
            metrics_dic = {
                'dice_loss': AverageMeter(),
                'dice_score': AverageMeter(),
                'iou': AverageMeter()
                }
            for ii, (images, labels, _) in enumerate(loader):
                images, labels = images.cuda(), labels.cuda()

                if split == 'train':
                    optimizer.zero_grad()
                    model.train()
                else:
                    model.eval()
                
                preds = model(images)
                loss = loss_fn(preds, labels)

                # Segmentation Metrics
                binary_preds = preds > 0.5
                binary_preds.float()
                dice_score = compute_dice_score(binary_preds,labels)
                iou = iou_coef(binary_preds,labels)

                if split == 'train':
                    loss.backward()
                    optimizer.step()
                #logging
                metrics_dic['dice_loss'].update(loss.item())
                metrics_dic['dice_score'].update(dice_score.mean().item())
                metrics_dic['iou'].update(iou.mean().item())


            saver.log_loss(f"{split}/loss",metrics_dic['dice_loss'].avg, epoch)
            saver.log_loss(f"{split}/dice_score",metrics_dic['dice_score'].avg, epoch)
            saver.log_loss(f"{split}/iou",metrics_dic['iou'].avg, epoch)

            if split == 'val' and metrics_dic['dice_loss'].avg < min_val_acc:
                min_val_acc = metrics_dic['dice_loss'].avg
                best_model = copy.deepcopy(model)
                min_val_epoch = epoch
            

    print('Training done.')


    ## save trained model
    saver.save_model(best_model,'best_model', min_val_epoch)
    print('Best Model saved.')

    #Save Validation Set predicitons (to run visualizations offline)
    os.makedirs(os.path.join(saver.path,'predictions'))
    for (img,label,idx) in val_loader_final:
        img = img.cuda()
        best_model.eval()

        with torch.no_grad():
            preds = best_model(img)
        np.save( os.path.join(saver.path,'predictions',f'pred_{idx.item()}') ,preds.cpu().squeeze().numpy())


if __name__ == "__main__":
    main()