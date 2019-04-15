import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import pandas as pd
import aimltools as aiml
import aimltools.io
from collections import defaultdict

''' See this blog for an example:

  https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
'''

def readClassListFile(fn):
    out = []
    with open(fn, 'r') as f:
        for line in f:
            l = line.strip()
            if len(l) > 0:
                out.append(l)
    return out


class GeneralDetection(data.Dataset):
    def __init__(self, root, classListFile, gtFileCSV, transform=None, datasetName='general'):
        self.root = root
        self.transform = transform
        self.name = datasetName
        # We need consistent class labels.  This is determined by the ordering in the class list file.
        self.classes = readClassListFile(os.path.join(self.root, classListFile))
        # Go from name to int index.
        self.classNameToIndex = dict(zip(self.classes, range(len(self.classes))))
        # Now read the actual data.
        self.df = pd.read_csv(os.path.join(self.root, gtFileCSV))
        # The data is one row per target object.  So multiple rows per image.
        # Turn this into indexable.
        self.filenameToArray = defaultdict(list)
        for row in self.df.iterrows():
            s = row[1]
            classIdx = self.classNameToIndex[s.classLabel]
            self.filenameToArray[s.filename].append([s.ulx, s.uly, s.ulx+s.w-1, s.uly+s.h-1, classIdx])

        filenames = list(self.filenameToArray.keys())
        self.indexToFilename = {}
        for i, fn in enumerate(filenames):
            # Build up a map from idx to filename.
            self.indexToFilename[i] = fn
            # And turn our list of targets into array
            self.filenameToArray[fn] = np.vstack(self.filenameToArray[fn]).astype(float)
            assert self.filenameToArray[fn].shape[1] == 5

    def __len__(self):
        return len(self.filenameToArray)

    def __getitem__(self, idx):
        fn = self.indexToFilename[idx]

        # Prepare image
        imgPath = os.path.join(self.root, fn)
        image = aimltools.io.readImage(imgPath)
        # If it's greyscale, interpret as RGB.
        if image.ndim == 2:
            image = np.dstack([image, image, image])
        image = image.astype(float)
        # Get target data.  Make our own copy to scale.
        target = np.array(self.filenameToArray[fn])

        H, W = image.shape[0], image.shape[1]
        # Scale to [0,1]
        target[:,0] /= float(W) #xmin
        target[:,1] /= float(H) #ymin
        target[:,2] /= float(W) #xmax
        target[:,3] /= float(H) #ymax

        # Apply transform
        if self.transform is not None:
            target = np.array(target)
            #print('debug: target = \n{}'.format(target))
            image, boxes, labels = self.transform(image, target[:, :4], target[:, 4])
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        # The target to be returned is an array of dimension (nbObjects, 5)
        # Each object target (row) is [xmin ymin xmax ymax classIndex]

        im = torch.from_numpy(image).permute(2, 0, 1).to(torch.float)
        gt = target
        #gt = torch.from_numpy(target).to(torch.float)
        #print('GENERAL getitem: im type, shape = {}, {}, gt type, dtype, shape = {}, {}, {}'.format(im.dtype, im.shape, type(gt), gt.dtype, gt.shape))

        assert np.all(aiml.inRange(gt[:, :4], 0, 1))
        return im, gt
