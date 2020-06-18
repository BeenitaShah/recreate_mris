import warnings
warnings.filterwarnings('ignore')
import h5py, os
from functions import transforms as T
from functions.subsample import MaskFunc
from scipy.io import loadmat
from torch.utils.data import DataLoader
import numpy as np
import torch
from matplotlib import pyplot as plt, animation as animation, style
from torch.nn import functional as F
import time
from unet_test import *
from torchsummary import summary
import gc
from skimage.measure import compare_ssim
import pandas as pd
import csv

def show_slices(data, slice_nums, cmap=None): # visualisation
    fig = plt.figure(figsize=(15,10))
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)
        plt.axis('off')

class MRIDataset(DataLoader):
    def __init__(self, data_list, acceleration, center_fraction, use_seed, is_undersampled):
        self.data_list = data_list
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.use_seed = use_seed
        self.is_undersampled = is_undersampled

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        subject_id = self.data_list[idx]
        return get_epoch_batch(subject_id, self.acceleration, self.center_fraction, self.is_undersampled, self.use_seed)

def get_epoch_batch(subject_id, acc, center_fract, is_undersampled, use_seed=True):
    ''' random select a few slices (batch_size) from each volume'''

    fname, rawdata_name, slice = subject_id  
    
    with h5py.File(rawdata_name, 'r') as data:
        if(is_undersampled):
            rawdata = data['kspace_8af'][slice]
        else:
            rawdata = data['kspace'][slice]
                      
    slice_kspace = T.to_tensor(rawdata).unsqueeze(0)
    S, Ny, Nx, ps = slice_kspace.shape

    # apply random mask
    shape = np.array(slice_kspace.shape)
    mask_func = MaskFunc(center_fractions=[center_fract], accelerations=[acc])
    seed = None if not use_seed else tuple(map(ord, fname))
    mask = mask_func(shape, seed)
      
    # undersample
    masked_kspace = torch.where(mask == 0, torch.Tensor([0]), slice_kspace)
    masks = mask.repeat(S, Ny, 1, ps)

    img_gt, img_und = T.ifft2(slice_kspace), T.ifft2(masked_kspace)

    # perform data normalization which is important for network to learn useful features
    # during inference there is no ground truth image so use the zero-filled recon to normalize
    norm = T.complex_abs(img_und).max()
    if norm < 1e-6: norm = 1e-6
    
    # normalized data
    img_gt, img_und, rawdata_und = img_gt/norm, img_und/norm, masked_kspace/norm
        
    return img_gt.squeeze(0), img_und.squeeze(0), rawdata_und.squeeze(0), masks.squeeze(0), norm

def load_data_path(train_data_path, val_data_path):
    #Go through each subset (training, validation) and list all
    #the file names, the file paths and the slices of subjects in the training and validation sets

    data_list = {}
    train_and_val = ['train', 'val']
    data_path = [train_data_path, val_data_path]
    #Leave list as sorted so the train,val and test sets will remain the same - i.e there won't be an overlab between them
    all_files = sorted(os.listdir(val_data_path))
    #Using 80% of data of total training - remaining 20% will be used for test
    all_files = all_files[:round(len(all_files)*0.8)]
    #Splitting up test set further into test and validation sets.
    #Using 70% of training data for training and 30% for validation
    train, val = all_files[:round(len(all_files)*0.7)] ,all_files[round(len(all_files)*0.7):]
    data_list['train'] = []
    data_list['val'] = []
    
    for index, fname in enumerate(train):
        data_set = 'train'
        which_data_path = train_data_path
        subject_data_path = os.path.join(which_data_path, fname) 
        
        if not os.path.isfile(subject_data_path): continue
        
        with h5py.File(subject_data_path, 'r') as data:
            num_slice = data['kspace'].shape[0]
            # the first 5 slices are mostly noise so it is better to exlude them
            data_list[data_set] += [(fname, subject_data_path, slice) for slice in range(5, num_slice)]
            
    for index, fname in enumerate(val):
        data_set = 'val'
        which_data_path = train_data_path
        subject_data_path = os.path.join(which_data_path, fname)
        
        if not os.path.isfile(subject_data_path): continue
            
        with h5py.File(subject_data_path, 'r') as data:
            num_slice = data['kspace'].shape[0]
            # the first 5 slices are mostly noise so it is better to exlude them
            data_list[data_set] += [(fname, subject_data_path, slice) for slice in range(5, num_slice)]
            
    return data_list 

#Data loading method used at test time
#Loads all slices from a single volume as a 'dataset' so can calculate averages over whole volume per image
def load_test_data_path(test_path, file_name):
    #Go through each subset (training, validation) and list all
    #the file names, the file paths and the slices of subjects in the training and validation sets
    data_list = {}
    data_list['test'] = [] 
    #print('Test path: ' + test_path)
    #print('File_name: ' + file_name)
    file_path = test_path + file_name
    #print('File path: ' + file_path)
    
    if os.path.isfile(file_path):
        with h5py.File(file_path, 'r') as data:
            num_slice = data['kspace'].shape[0]
            # the first 5 slices are mostly noise so it is better to exlude them
            #print('Number of Slices: ' + str(num_slice))
            data_list['test'] += [(file_name, file_path, slice) for slice in range(5, num_slice)]
            
    return data_list  

#Data loading method used at test time
#Loads all slices from a single volume as a 'dataset' so can calculate averages over whole volume per image
def load_undersampled_data_path(test_path, file_name):
    #Go through each subset (training, validation) and list all
    #the file names, the file paths and the slices of subjects in the training and validation sets
    data_list = {}
    data_list['test'] = [] 
    #print('Test path: ' + test_path)
    #print('File_name: ' + file_name)
    file_path = test_path + file_name
    #print('File path: ' + file_path)
    
    if os.path.isfile(file_path):
        with h5py.File(file_path, 'r') as data:
            num_slice = data['kspace_8af'].shape[0]
            # the first 5 slices are mostly noise so it is better to exlude them
            #print('Number of Slices: ' + str(num_slice))
            data_list['test'] += [(file_name, file_path, slice) for slice in range(5, num_slice)]
            
    return data_list  

from skimage.measure import compare_ssim 
def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return compare_ssim(
        gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
    )

#Method called once per epoch during training
def train_epoch(model, optimizer, train_loader):
    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    for iteration, sample in enumerate(train_loader):
        
        img_gt, img_und, rawdata_und, masks, norm = sample
         
        # Extract input and ground truth image
        input_img = T.complex_abs(img_und).squeeze()
        input_img = T.center_crop(input_img, [320, 320])
        input_img = input_img[None,None,:,:].cuda()
        target_img = T.complex_abs(img_gt).squeeze()
        target_img = T.center_crop(target_img, [320, 320])
        target_img = target_img[None,None,:,:].cuda()
        
        output = model(input_img)
        loss = F.l1_loss(output, target_img)
        #loss = ssim(target_img, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iteration > 0 else loss.item()
    
    return avg_loss, time.perf_counter() - start_epoch

#Method called once per epoch during training, after train_epoch has been called.
#Evaluates how well model is performing
def val_epoch(model, test_loader):
    #Setting model to evaluation mode
    model.eval()
    start_epoch = start_iter = time.perf_counter()
    losses = []
    
    for iteration, sample in enumerate(test_loader):
        img_gt, img_und, rawdata_und, masks, norm = sample
         
        # Extract input and ground truth image
        input_img = T.complex_abs(img_und).squeeze()
        input_img = T.center_crop(input_img, [320, 320])
        input_img = input_img[None,None,:,:].cuda()
        target_img = T.complex_abs(img_gt).squeeze()
        target_img = T.center_crop(target_img, [320, 320])
        target_img = target_img[None,None,:,:].cuda()
        
        output = model(input_img)
        loss = F.l1_loss(output, target_img)
        losses.append(loss.item())

    return np.mean(losses), time.perf_counter() - start_epoch
    
#Called when evaluating test set.
#Called once for every FILE
def test_epoch(model, test_loader):
    model.eval()
    start_epoch = start_iter = time.perf_counter()
    losses = []
    output_vol = []
    gt_vol = []
    
    for iteration, sample in enumerate(test_loader):
        img_gt, img_und, rawdata_und, masks, norm = sample
         
        # Extract input and ground truth image
        input_img = T.complex_abs(img_und).squeeze()
        input_img = T.center_crop(input_img, [320, 320])
        input_img = input_img[None,None,:,:].cuda()
        
        output = model(input_img)
        output = output[0,0,:,:].cpu().detach()
        output_vol.append(output*norm)
        
    #return np.mean(losses), time.perf_counter() - start_epoch
    return output_vol, time.perf_counter() - start_epoch

#Matplotlib update method
def mpl_update():    
    # TODO only show last N points, or will slowly hog memory & processing power
    line1.set_ydata(np.asarray(train_loss_ys))
    line2.set_ydata(np.asarray(val_loss_ys))
    line1.set_xdata(np.asarray(epoch_list))
    line2.set_xdata(np.asarray(epoch_list))
    ax.set_xlim(0, max(epoch_list))
    ax.set_ylim(0, max([max(train_loss_ys),max(val_loss_ys)]))
    fig.canvas.draw()
    fig.canvas.flush_events()

def eval_model(model):
    #Testing the trained model on test set - 20%
    data_path = '/data/local/NC2019MRI/train/'
    all_files = sorted(os.listdir(data_path))
    #Using 20% of data as test set
    all_files = all_files[round(len(all_files)*0.8):]

    #acc and cent_fract should be (8, 0.04) or (4, 0.08)
    acc = 8
    cen_fract = 0.04
    # random masks for each slice 
    seed = False
    # data loading is faster using a bigger number for num_workers. 0 means using one cpu to load data
    num_workers = 12 

    all_losses = []

    for file in all_files:
        # first load all file names, paths and slices.
        data_list = load_test_data_path(data_path, file)
        #Create a dataloader containing all slices for particular slice
        test_dataset = MRIDataset(data_list['test'], acceleration=acc, center_fraction=cen_fract, use_seed=seed,is_undersampled=False)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=num_workers)
        #Call model over all slices. pred_vol is a list of volume slices (in correct order)
        pred_vol,test_time = test_epoch(model, test_loader)
        
        #3d Volume achieved by stacking slices
        pred = torch.stack(pred_vol, dim=0)
        
        #Loading Ground Truth
        with h5py.File(data_path + file,  "r") as hf:
            volume_kspace = hf['kspace'][()]
            
        #Transform k-space to real valued image
        volume_kspace2 = T.to_tensor(volume_kspace)
        # Apply Inverse Fourier Transform to get the complex image
        volume_image = T.ifft2(volume_kspace2)            
        volume_image_abs = T.complex_abs(volume_image)
        volume_image_abs = volume_image_abs[5:, :, :]
        real_gt = T.center_crop(volume_image_abs, [320, 320])
        
        #visualize predicted and gt slices
        show_slices(pred, [5, 10, 20, pred.shape[0]-1], cmap='gray')
        show_slices(real_gt, [5, 10, 20, pred.shape[0]-1], cmap='gray')
        
        #calculate ssim score between volumes
        ssim_score = ssim(real_gt.numpy(), pred.numpy())
        all_losses.append(ssim_score)
    
    average_ssim = sum(all_losses) / len(all_losses)
    return average_ssim

def main():
    data_path_train = '/data/local/NC2019MRI/train'
    data_path_val = '/data/local/NC2019MRI/train'
    data_list = load_data_path(data_path_train, data_path_val) # first load all file names, paths and slices.

    acc = 8
    cen_fract = 0.04
    seed = False # random masks for each slice 
    num_workers = 12 # data loading is faster using a bigger number for num_workers. 0 means using one cpu to load data
    
    # create data loader for training set. It applies same to validation set as well
    train_dataset = MRIDataset(data_list['train'], acceleration=acc, center_fraction=cen_fract, use_seed=seed,is_undersampled=False)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=num_workers)
    
    val_dataset = MRIDataset(data_list['val'], acceleration=acc, center_fraction=cen_fract, use_seed=seed,is_undersampled=False)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=1, num_workers=num_workers)

    parameter_training_info = pd.DataFrame(columns=['Epoch', 'Average_ssim', 'TrainingLoss', 'TrainingTime', 'ValidationLoss', 'ValidationTime'])
    #Change name of file to match optimser you are testing
    file_name = 'final_epochs.csv'
    parameter_training_info.to_csv(file_name)

    model = UnetModel(in_chans=1, out_chans=1, chans=64, num_pool_layers=4, drop_prob = 0.05)
    model.cuda()
    train_loss_ys = []
    training_time = []
    val_loss_ys = []
    validation_time = []

    #Change name of optimiser here
    optimizer = torch.optim.RMSprop(model.parameters(), 0.0003)
    for e in range(150):
        print('Epoch: ' + str(e))
        train_loss, train_time = train_epoch(model, optimizer, train_loader)
        val_loss, val_time = val_epoch(model, val_loader)
        #Metrics update
        train_loss_ys.append(train_loss)
        training_time.append(train_time)
        val_loss_ys.append(val_loss)
        validation_time.append(val_time)
        average_ssim = eval_model(model)
        print('Writing to file')
        with open(file_name,'a') as f:
            writer=csv.writer(f)
            writer.writerow([str(e), str(average_ssim), [train_loss_ys], [training_time], [val_loss_ys], [validation_time]])
            gc.collect()
    gc.collect()
            

if __name__ == '__main__':
    main()
