import numpy as np
import os
import h5py
from torch.utils.data import Dataset
from scipy.fft import fft
from scipy import signal
from scipy.signal import butter, filtfilt
import random

def UBFC_LU_split():
    # split UBFC dataset into training and testing parts
    # the function returns the file paths for the training set and test set.
    # TODO: if you want to train on another dataset, you should define new train-test split function.
    
    h5_dir = '../datasets/UBFC_h5'
    train_list = []
    val_list = []

    val_subject = [49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38]

    for subject in range(1,50):
        if os.path.isfile(h5_dir+'/%d.h5'%(subject)):
            if subject in val_subject:
                val_list.append(h5_dir+'/%d.h5'%(subject))
            else:
                train_list.append(h5_dir+'/%d.h5'%(subject))

    return train_list, val_list    

# pure dataset split method defined by czy
def PURE_split():
    # split PURE dataset into training and testing parts
    # the function returns the file paths for the training set and test set.
    
    h5_dir = '../drive/MyDrive/'
    train_list = []
    val_list = []
    
    subject_list = ['01-01lmout.h5','01-02lmout.h5','01-03lmout.h5','01-04lmout.h5','01-05lmout.h5','01-06lmout.h5','02-01lmout.h5','02-02lmout.h5','02-03lmout.h5','02-04lmout.h5','02-05lmout.h5','02-06lmout.h5','03-01lmout.h5','03-02lmout.h5','03-03lmout.h5','03-04lmout.h5','03-05lmout.h5','03-06lmout.h5','04-01lmout.h5','04-02lmout.h5','04-03lmout.h5','04-04lmout.h5','04-05lmout.h5','04-06lmout.h5','05-01lmout.h5','05-02lmout.h5','05-03lmout.h5','05-04lmout.h5','05-05lmout.h5','05-06lmout.h5','06-01lmout.h5','06-03lmout.h5','06-04lmout.h5','06-05lmout.h5','06-06lmout.h5','07-01lmout.h5','07-02lmout.h5','07-03lmout.h5','07-04lmout.h5','07-05lmout.h5','07-06lmout.h5','08-01lmout.h5','08-02lmout.h5','08-03lmout.h5','08-04lmout.h5','08-05lmout.h5','08-06lmout.h5','09-01lmout.h5','09-02lmout.h5','09-03lmout.h5','09-04lmout.h5','09-05lmout.h5','09-06lmout.h5','10-01lmout.h5','10-02lmout.h5','10-03lmout.h5','10-04lmout.h5','10-05lmout.h5','10-06lmout.h5']
    random.seed(17)
    train_list = random.sample(subject_list, k=int(0.8 * len(subject_list)))
    val_list = list(set(subject_list) - set(train_list))
    train_list = [h5_dir + i  for i in train_list]
    val_list = [h5_dir + i  for i in val_list]

    return train_list, val_list  
    
class H5Dataset(Dataset):

    def __init__(self, train_list, T):
        self.train_list = train_list # list of .h5 file paths for training
        self.T = T # video clip length

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        with h5py.File(self.train_list[idx], 'r') as f:
            img_length = f['imgs'].shape[0]
            # 为什么要随机选取一定长度的视频而不是把视频分段？
            idx_start = np.random.choice(img_length-self.T)

            idx_end = idx_start+self.T

            img_seq = f['imgs'][idx_start:idx_end]
            img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32')
        return img_seq

class H5Dataset_with_landmark(Dataset):

    def __init__(self, train_list, T):
        self.train_list = train_list # list of .h5 file paths for training
        self.T = T # video clip length

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        with h5py.File(self.train_list[idx], 'r') as f:
            img_length = f['imgs'].shape[0]
            
            # 为什么要随机选取一定长度的视频而不是把视频分段？
            idx_start = np.random.choice(img_length-self.T)

            idx_end = idx_start+self.T

            img_seq = f['imgs'][idx_start:idx_end]
            land_mark_seq = f['landmarks'][idx_start:idx_end]
            img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32')
        return img_seq,land_mark_seq
