import numpy as np
import os
import h5py
from torch.utils.data import Dataset
from scipy.fft import fft
from scipy import signal
from scipy.signal import butter, filtfilt

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
    
    subject_list = ['0'+str(i+1) + '-0' + str(j+1)  for i in range(4) for j in range(6)]
    # 17 subjects for validation
    train_list = ["01-03", "01-01", "04-05", "02-04", "04-02", "03-06", "04-03", "02-03", "04-04", "02-05", "02-02", "02-06", "02-01", "01-04", "01-02", "03-03", "03-02",'05-01','05-02','05-04','05-06']
    
    val_list = [subject for subject in subject_list if subject not in train_list]
    train_list = [h5_dir + i +'lmout.h5' for i in train_list]
    val_list = [h5_dir + i +'lmout.h5' for i in val_list]

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
