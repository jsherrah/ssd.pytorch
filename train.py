from data import *
from utils.augmentations import * #SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
from torchsummary import summary
import torchvision
from torchvision import transforms
import numpy as np
import argparse
import collections
import json
import matplotlib.pyplot as plt

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
#train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO', 'general'], type=str, help='VOC or COCO or general')
parser.add_argument('--dataset_root', default=VOC_ROOT, help='Dataset root directory path')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--weightsFile', default=None, type=str, help='File to load weights from.')
parser.add_argument('--loadMode', choices=['resume', 'finetune', 'basenet', 'none'], default='basenet', type=str, help='How to interpret weightsFile.\n\tfinetune: Finetune the given net, so all aspects might match, except for latter layers.\n\tbasenet: Pretrained base model for features.\n\tresume: weights must match exactly.')
parser.add_argument('--start_iter', default=0, type=int, help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/', help='Directory for saving checkpoint models')
parser.add_argument('--loadExtras', action='store_true', help='If true, load non-feature layers')

parser.add_argument('--displayIters', default=10, type=int, help='Display every this many iterations')
parser.add_argument('--saveIters', default=1000, type=int, help='Save every this many iterations')
parser.add_argument('--testIters', default=50, type=int, help='Test every this many iterations')

parser.add_argument('--configFile', type=str, default=None, help='json config file for data set')
parser.add_argument('--classListFile', type=str, default=None, help='file')
parser.add_argument('--gtFileCSV', type=str, default=None, help='file')
parser.add_argument('--dataName', type=str, default=None, help='file')

parser.add_argument('--dataset_rootTest', default=None, help='Dataset root directory path')
parser.add_argument('--gtFileCSVTest', type=str, default=None, help='file')
parser.add_argument('--nbBatchesTest', default=10, type=int, help='Number of batches to run each test iteration')

args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def readConfigFile(fn):
    with open(fn, 'r') as f:
        cfg = json.load(f)
    return cfg



def imshowTensor(inp, title=None, means=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    if means is not None:
        #print('adding means')
        # Because at this point it's rgb
        inp += means[::-1]
    inp = np.clip(inp, 0, 255)
    #print('inp type = {}, min={}, max={}'.format(inp.dtype, inp.min(), inp.max()))
    plt.imshow(inp.astype(np.uint8))
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def showDataSet(trainLoader, means):
    # inspired by https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    class_names = trainLoader.dataset.classes
    # Get a batch of training data
    inputs, classes = next(iter(trainLoader))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imshowTensor(out, means=means)


def train():
    waysAndMeans = None

    datasetTest = None

    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        waysAndMeans = MEANS
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'], waysAndMeans))
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        waysAndMeans = MEANS
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'], waysAndMeans))
    elif args.dataset == 'general':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset "general"')
        cfg = readConfigFile(args.configFile)
        print('Using general data set with config = \n{}'.format(cfg))
        waysAndMeans = cfg['means']
        transformer = SSDAugmentation(cfg['min_dim'], waysAndMeans, [
            ConvertFromInts(),
            #ToAbsoluteCoords(),
            #PhotometricDistort(),
            #Expand(self.mean),
            #RandomSampleCrop(),
            #RandomMirror(),
            #ToPercentCoords(),
            #Resize(self.size),
            SubtractMeans(waysAndMeans)
        ])
        dataset = GeneralDetection(args.dataset_root,
                                   args.classListFile,
                                   args.gtFileCSV,
                                   transform=transformer,
                                   datasetName=args.dataName)

        if args.dataset_rootTest is not None and args.gtFileCSVTest is not None:
            print('Loading test set')
            transformerTest = SSDAugmentation(cfg['min_dim'], waysAndMeans, [
                ConvertFromInts(),
                #ToAbsoluteCoords(),
                #PhotometricDistort(),
                #Expand(self.mean),
                #RandomSampleCrop(),
                #RandomMirror(),
                #ToPercentCoords(),
                #Resize(self.size),
                SubtractMeans(waysAndMeans)
            ])
            datasetTest = GeneralDetection( args.dataset_rootTest,
                                            args.classListFile,
                                            args.gtFileCSVTest,
                                            transform=transformerTest,
                                            datasetName=args.dataName )

    imgSz = cfg['min_dim']
    print('  - done')

    if args.visdom:
        print('setting up visdom')
        import visdom
        viz = visdom.Visdom()
        print('  - done')
    else:
        viz = None

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    # Here is where we load the weights.  It can get very confusing.
    # TO help, print out keys of state dict first.
    print('Created neural net with state dict = \n{}'.format(ssd_net.state_dict().keys()))

    if args.weightsFile is None or args.loadMode == 'none':
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)
    else:
        wtsLoaded = torch.load(args.weightsFile)
        print('Loading file with State Dict Keys = \n{}'.format(wtsLoaded.keys()))

        if args.loadMode == 'resume':
            print('Resuming training, loading {}...'.format(args.weightsFile))
            ssd_net.load_weights(args.weightsFile)
        elif args.loadMode == 'basenet':
            ssd_net.vgg.load_state_dict(wtsLoaded)
        elif args.loadMode == 'finetune':
            # OK if we have too many layers on input
            wtsFiltered = collections.OrderedDict()
            for k, v in wtsLoaded.items():
                if k.startswith('vgg.') or k.startswith('extras.'):
                    wtsFiltered[k] = v
            print('Fine-tuning model, just loading these weights:\n{}'.format(wtsFiltered.keys()))
            ssd_net.load_state_dict(wtsFiltered, strict=False)
        else:
            print('Invalid load mode {}'.format(args.loadMode))
            sys.exit(1)

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.cuda:
        net = net.cuda()

    print('Loaded model = ')
    summary(ssd_net, (3, cfg['min_dim'], cfg['min_dim']), device='cuda' if args.cuda else 'cpu')

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)
    print('%d iterations per epoch' % epoch_size)
    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot(viz, 'Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot(viz, 'Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    if datasetTest is None:
        data_loaderTest = None
        iter_plotTest = None
    else:
        data_loaderTest = data.DataLoader(datasetTest, args.batch_size,
                                          num_workers=args.num_workers,
                                          shuffle=True, collate_fn=detection_collate,
                                          pin_memory=True)
        iter_plotTest = create_vis_plot(viz, 'Iteration', 'Loss',  'SSD.PyTorch on ' + dataset.name + ' - test', vis_legend)
        epoch_plotTest = create_vis_plot(viz, 'Epoch', 'Loss', 'SSD.PyTorch on ' + dataset.name + ' - test', vis_legend)
        epoch_sizeTest = len(datasetTest) // args.batch_size

    plt.ion()
    showDataSet(data_loader, means=waysAndMeans)
    #plt.waitforbuttonpress()

    # create batch iterator
    batch_iterator = iter(data_loader)
    batch_iterator_test = None
    for iteration in range(args.start_iter, cfg['max_iter']):

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration as e:
            # End of an epoch:
            #   no more batches left, start again
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

            # Evaluate on whole test set.
            print('End of epoch %3d ||' % epoch, end=' ')
            if data_loaderTest is not None:
                _, lossTest_l, lossTest_c = validate(
                    data_loaderTest, net, criterion, args.cuda, None, epoch_sizeTest, doPrint=True
                )
                net.train()
            print('')

            if args.visdom:
                loc_loss /= float(epoch_size)
                conf_loss /= float(epoch_size)
                update_vis_plot(viz, epoch, loc_loss, conf_loss, epoch_plot, 'append')
                update_vis_plot(viz, epoch, lossTest_l, lossTest_c, epoch_plotTest, 'append')

            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        #print('targets = {}'.format(targets))
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        #print('out  = \n{}, targets = \n{}'.format(out, targets))
        loss_l, loss_c = criterion(out, targets)
        #print('loss_l = {}, loss_c = {}'.format(loss_l, loss_c))
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        if iteration % args.displayIters == 0:
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

            if args.visdom:
                update_vis_plot(viz, iteration, loss_l.item(), loss_c.item(), iter_plot, 'append')

        if data_loaderTest is not None and args.nbBatchesTest > 0 and iteration % args.testIters == 0:
            # Test
            batch_iterator_test, lossTest_l, lossTest_c = validate(
                data_loaderTest, net, criterion, args.cuda, batch_iterator_test, args.nbBatchesTest, doPrint=True
            )
            net.train()

            if args.visdom:
                update_vis_plot(viz, iteration, lossTest_l, lossTest_c, iter_plotTest, 'append')

        if iteration % args.displayIters == 0:
            print('timer: %.4f sec.' % (t1 - t0))

        if iteration != 0 and iteration % args.saveIters == 0:
            print('Saving state, iter:', iteration)
            if args.dataName is None:
                dataName = arts.dataset
            else:
                dataName = args.dataName
            ofn = '%s/ssd%d_%s_%08d.pth' % (args.save_folder, imgSz, dataName, iteration)
            torch.save(ssd_net.state_dict(), ofn)

    torch.save(ssd_net.state_dict(), '%s/ssd%d_%s_%08d.pth' % (args.save_folder, imgSz, dataName, iteration))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(viz, _xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(viz, iteration, loc, conf, window1, update_type):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
        win=window1,
        update=update_type
    )

################################3
def validate(data_loader, net, criterion, cuda, batch_iterator, nbBatches, doPrint):
    if batch_iterator is None:
        batch_iterator = iter(data_loader)

    net.eval()
    loss_l_tot, loss_c_tot = 0, 0
    count = 0

    with torch.no_grad():
        for iteration in range(nbBatches):
            # load train data
            try:
                images, targets = next(batch_iterator)
            except StopIteration as e:
                # no more batches left, start again
                print('Restarting validation batch iterator')
                batch_iterator = iter(data_loader)
                images, targets = next(batch_iterator)

            if cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda()) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann) for ann in targets]

            out = net(images)
            loss_l, loss_c = criterion(out, targets)
            loss_l_tot += loss_l.item()
            loss_c_tot += loss_c.item()
            count += 1

    loss_l_tot /= float(count)
    loss_c_tot /= float(count)
    loss = loss_l_tot + loss_c_tot

    if doPrint:
        # This adds to existing line
        print(' Test Loss: %.4f ||' % (loss), end=' ')

    return batch_iterator, loss_l_tot, loss_c_tot


if __name__ == '__main__':
    train()
