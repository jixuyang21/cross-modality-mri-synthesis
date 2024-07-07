import os
from glob import glob
import nibabel as nib
import numpy as np

from torch.utils.data import Dataset

class BratsDataset(Dataset):
    def __init__(self, root, transforms, mode, pad):
        self.root = os.path.join(root, mode)
        self.subjects = sorted(glob(os.path.join(self.root, "*")))
        self.transforms = transforms
        self.modalities = ['t1', 't2', 't1ce', 'flair']
        self.pad = pad

    def __getitem__(self, index):
        modal_dict = {}
        idx = (index // 10) + 1
        idx = (3 - len(str(idx))) * '0' + str(idx)
        slice_idx = 75 + index % 10
        
        for modality in self.modalities:
            path = os.path.join(self.root, idx, f"{modality}.nii.gz")
            img = nib.load(path).get_fdata()[:, :, slice_idx:slice_idx+1]
            
            img = (img - img.min()) / (img.max() - img.min())
            if self.pad:
                img = np.pad(img, ((8, 8), (8, 8), (0, 0)), mode='constant', constant_values=0)
            if self.transforms:
                img = self.transforms(img)

            modal_dict[modality] = img
        
        return modal_dict

    def __len__(self):
        return len(self.subjects) * 10
    