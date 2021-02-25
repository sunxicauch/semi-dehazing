import time
import util.util as util
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from util.metrics import PSNR
from util.metrics import SSIM
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")

opt = TestOptions().parse()
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batch_size = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip


model = create_model(opt)

class TestDataset(Dataset):
    def __init__(self, root='./'):
        self.root = root
        self.image_paths = glob.glob(self.root + '/*.png')
        self.transform1 = transforms.Compose([transforms.ToTensor()])
        self.transform2 = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))])
        
    def __len__(self):
        return len(glob.glob(self.root+'/*.png'))
    
    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert('RGB')
        A = self.transform1(img)
        A = self.transform2(A)
        return {'A': A, 'A_paths': self.image_paths[index], 'B': A, 'C': A}        

dataset = TestDataset(opt.dataroot)
loader = DataLoader(dataset, batch_size=1)
print(len(dataset))
    
with torch.no_grad():
    for i, data in enumerate(loader):
        model.set_input(data)
        model.test()
        img_path = model.get_image_paths()
        results = util.tensor2im(model.fake_B)
        print(results.shape)
        plt.imsave('rst{}.png'.format(i), results)
        
        
    
