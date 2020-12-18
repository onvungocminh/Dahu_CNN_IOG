from datetime import datetime
import scipy.misc as sm
from collections import OrderedDict
import glob
import numpy as np
import socket
import timeit

# PyTorch includes
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

# Custom includes
from dataloaders.combine_dbs import CombineDBs as combine_dbs
import dataloaders.pascal as pascal
import dataloaders.sbd as sbd
from dataloaders import custom_transforms_dahu as tr
from dataloaders.helpers import *
from networks.loss import class_cross_entropy_loss
from networks.mainnetwork import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2


class _RepeatSampler(object):

    """ Sampler that repeats forever.
    Args:
    sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)



class FastDataLoader(torch.utils.data.dataloader.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler',  _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


# Set gpu_id to -1 to run in CPU mode, otherwise set the id of the corresponding gpu
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('Using GPU: {} '.format(gpu_id))

# Setting parameters
use_sbd = True # train with SBD
nEpochs = 100  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
p = OrderedDict()  # Parameters to include in report
p['trainBatch'] = 3  # Training batch size 5
p['valBatch'] = 3 # Training batch size 5
snapshot = 1  # Store a model every snapshot epochs
nInputChannels = 6  # Number of input channels (RGB + heatmap of extreme points)
p['nAveGrad'] = 1  # Average the gradient of several iterations
p['lr'] = 1e-8  # Learning rate
p['wd'] = 0.0005  # Weight decay
p['momentum'] = 0.9  # Momentum

# Results and model directories (a new directory is generated for every run)
save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
if resume_epoch == 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
else:
    run_id = 0
save_dir = os.path.join(save_dir_root, 'run' )
if not os.path.exists(os.path.join(save_dir, 'models')):
    os.makedirs(os.path.join(save_dir, 'models'))

# Network definition
modelName = 'IOG_pascal'
net = Network(nInputChannels=nInputChannels,num_classes=1,
                        backbone='resnet101',
                        output_stride=16,
                        sync_bn=None,
                        freeze_bn=False,
                        pretrained=False)
if resume_epoch == 0:
    print("Initializing from pretrained model")
else:
    print("Initializing weights from: {}".format(
        os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
    net.load_state_dict(
        torch.load(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'),
                   map_location=lambda storage, loc: storage))

train_params = [{'params': net.get_1x_lr_params(), 'lr': p['lr']},
                {'params': net.get_10x_lr_params(), 'lr': p['lr'] * 10}]
net.to(device)

if resume_epoch != nEpochs:
    # Logging into Tensorboard
    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())

    # Use the following optimizer
    optimizer = optim.SGD(train_params, lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])
    p['optimizer'] = str(optimizer)

    # Preparation of the data loaders
    composed_transforms_tr = transforms.Compose([
        # tr.RGBShift(shift_limit=10, img_elem='image'),
        # tr.RandomRotate90(),
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-20, 20), scales=(.7, 1.1)),
        tr.CropFromMask(crop_elems=('image', 'gt','void_pixels'), relax=30, zero_pad=True),
        tr.FixedResize(resolutions={'crop_image': (512, 512), 'crop_gt': (512, 512), 'crop_void_pixels': (512, 512)},flagvals={'crop_image':cv2.INTER_LINEAR,'crop_gt':cv2.INTER_LINEAR,'crop_void_pixels': cv2.INTER_LINEAR}),
        tr.IOGPoints(sigma=10, img_elem = 'crop_image',elem='crop_gt',pad_pixel=10),
        tr.ToImage(norm_elem='IOG_points'),
        tr.ConcatInputs(elems=('crop_image', 'IOG_points')),
        tr.ToTensor()])

    composed_transforms_ts = transforms.Compose([
        tr.CropFromMask(crop_elems=('image', 'gt','void_pixels'), relax=30, zero_pad=True),
        tr.FixedResize(resolutions={'crop_image': (512, 512), 'crop_gt': (512, 512), 'crop_void_pixels': (512, 512)},flagvals={'crop_image':cv2.INTER_LINEAR,'crop_gt':cv2.INTER_LINEAR,'crop_void_pixels': cv2.INTER_LINEAR}),
        tr.IOGPoints(sigma=10, img_elem = 'crop_image', elem='crop_gt',pad_pixel=10),
        tr.ToImage(norm_elem='IOG_points'),
        tr.ConcatInputs(elems=('crop_image', 'IOG_points')),
        tr.ToTensor()])

    voc_train = pascal.VOCSegmentation(split='train', transform=composed_transforms_tr)
    voc_val = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts)
    if use_sbd:
        sbd = sbd.SBDSegmentation(split=['train', 'val'], transform=composed_transforms_tr, retname=True)
        db_train = combine_dbs([voc_train, sbd], excluded=[voc_val])
        db_val = voc_val

    else:
        db_train = voc_train
        db_val = voc_val

    p['dataset_train'] = str(db_train)
    p['transformations_train'] = [str(tran) for tran in composed_transforms_tr.transforms]
    # trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True,  drop_last=True)
    # valloader = DataLoader(db_val, batch_size=p['valBatch'], shuffle=False,  drop_last=True)
    trainloader = FastDataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, drop_last=True)
    valloader = FastDataLoader(db_val, batch_size=p['valBatch'], shuffle=False, drop_last=True)


    # Train variables
    num_img_tr = len(trainloader)
    running_loss_tr = 0.0
    num_img_ts = len(valloader)
    val_loss_tr = 0.0
    aveGrad = 0
    print("Training Network")

    writer = SummaryWriter()

    for epoch in range(resume_epoch, nEpochs):
        start_time = timeit.default_timer()
        epoch_loss = []


        # train dataset
        with tqdm(total=num_img_tr, desc=f'Epoch{epoch + 1} / {nEpochs}', unit='it') as pbar:
            for ii, sample_batched in enumerate(trainloader):
                gts =  sample_batched['crop_gt']
                inputs = sample_batched['concat']
                void_pixels =  sample_batched['crop_void_pixels']
                metas = sample_batched['meta']


                # y1 = torch.Tensor.cpu(gts).detach().numpy()[0, 0, :, :]
                # y2 = torch.Tensor.cpu(inputs).detach().numpy()[0, 5, :, :]
                # y4 = torch.Tensor.cpu(inputs).detach().numpy()[0, 0, :, :]
                # y5 = torch.Tensor.cpu(inputs).detach().numpy()[0, 1, :, :]
                # y6 = torch.Tensor.cpu(inputs).detach().numpy()[0, 2, :, :]
                #
                # tmp = np.zeros((512,512,3), 'uint8')
                # tmp[:,:,0] = y4
                # tmp[:,:,1] = y5
                # tmp[:,:,2] = y6
                #
                # y8 = y2/(np.max(y2) + 0.001) *255
                # y3 = y1 *255
                #
                #
                #
                # cv2.imwrite('/media/minh/MEDIA/Study/deeplearning/interactive_segmentation/iog/dahu/run/Results/'+ metas['image'][0] + '-' + metas['object'][0] + '.png', np.asarray(y8,dtype=np.uint8))
                # cv2.imwrite('/media/minh/MEDIA/Study/deeplearning/interactive_segmentation/iog/dahu/run/GT/'+ metas['image'][0] + '-' + metas['object'][0] + '.png', np.asarray(y3,dtype=np.uint8))
                # cv2.imwrite('/media/minh/MEDIA/Study/deeplearning/interactive_segmentation/iog/dahu/run/GT/'+ metas['image'][0] + '-' + metas['object'][0] + '_or.png', np.asarray(tmp,dtype=np.uint8))




                net.train()
                inputs.requires_grad_()
                inputs, gts ,void_pixels = inputs.to(device), gts.to(device), void_pixels.to(device)
                coarse_outs1,coarse_outs2,coarse_outs3,coarse_outs4,fine_out = net.forward(inputs)

                # Compute the losses
                loss_coarse_outs1 = class_cross_entropy_loss(coarse_outs1, gts, void_pixels=void_pixels)
                loss_coarse_outs2 = class_cross_entropy_loss(coarse_outs2, gts, void_pixels=void_pixels)
                loss_coarse_outs3 = class_cross_entropy_loss(coarse_outs3, gts, void_pixels=void_pixels)
                loss_coarse_outs4 = class_cross_entropy_loss(coarse_outs4, gts, void_pixels=void_pixels)
                loss_fine_out = class_cross_entropy_loss(fine_out, gts, void_pixels=void_pixels)
                loss = loss_coarse_outs1+loss_coarse_outs2+ loss_coarse_outs3+loss_coarse_outs4+loss_fine_out

                # if ii % 10 ==0:
                #     print('Epoch',epoch,'step',ii,'loss',loss)
                running_loss_tr += loss.item()

                # Print stuff
                if ii % num_img_tr == num_img_tr - 1 -p['trainBatch']:
                    running_loss_tr = running_loss_tr / num_img_tr
                    train_loss = running_loss_tr
                    print('[Epoch: %d, numImages: %5d]' % (epoch, ii*p['trainBatch']+inputs.data.shape[0]))
                    print('Loss: %f' % train_loss)
                    running_loss_tr = 0
                    stop_time = timeit.default_timer()
                    print("Execution time: " + str(stop_time - start_time)+"\n")

                # Backward the averaged gradient
                loss /= p['nAveGrad']
                loss.backward()
                aveGrad += 1
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(1)

                # Update the weights once in p['nAveGrad'] forward passes
                if aveGrad % p['nAveGrad'] == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    aveGrad = 0

        # Save the model
        if (epoch % snapshot) == snapshot - 1 and epoch != 0:
            torch.save(net.state_dict(), os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth'))


        # val dataset
        for ii, sample_batched in enumerate(valloader):
            gts =  sample_batched['crop_gt']
            inputs = sample_batched['concat']
            void_pixels =  sample_batched['crop_void_pixels']
            # inputs.requires_grad_()
            inputs, gts ,void_pixels = inputs.to(device), gts.to(device), void_pixels.to(device)
            with torch.no_grad():
                coarse_outs1,coarse_outs2,coarse_outs3,coarse_outs4,fine_out = net.forward(inputs)

            # Compute the losses
            loss_coarse_outs1 = class_cross_entropy_loss(coarse_outs1, gts, void_pixels=void_pixels)
            loss_coarse_outs2 = class_cross_entropy_loss(coarse_outs2, gts, void_pixels=void_pixels)
            loss_coarse_outs3 = class_cross_entropy_loss(coarse_outs3, gts, void_pixels=void_pixels)
            loss_coarse_outs4 = class_cross_entropy_loss(coarse_outs4, gts, void_pixels=void_pixels)
            loss_fine_out = class_cross_entropy_loss(fine_out, gts, void_pixels=void_pixels)
            val_loss = loss_coarse_outs1+loss_coarse_outs2+ loss_coarse_outs3+loss_coarse_outs4+loss_fine_out


            val_loss_tr += val_loss.item()

            # Print stuff
            if ii % num_img_ts == num_img_ts - 1 -p['valBatch']:
                val_loss_tr = val_loss_tr / num_img_ts
                val_loss_total = val_loss_tr
                print('[Epoch: %d, numImages: %5d]' % (epoch, ii*p['valBatch']+inputs.data.shape[0]))
                print('Loss: %f' % val_loss_total)
                val_loss_tr = 0
                stop_time = timeit.default_timer()
                print("Execution time: " + str(stop_time - start_time)+"\n")


        writer.add_scalars('loss/train/val/', {'loss/train':train_loss,
                                        'loss/val': val_loss_total}, epoch)
