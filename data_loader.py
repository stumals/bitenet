import torch
import numpy as np

from data_prep import data_prep, get_last_code, dataset_encoding, target_to_onehot
from torch.utils.data import Dataset, TensorDataset


class MedDataset(Dataset):
     def __init__(self, max_visits, datasets_path, admin_file, diag_file, procedure_file, prescript_file, drug_file):
        data, icd9_diag_categories = data_prep(datasets_path, admin_file, diag_file, procedure_file, prescript_file, drug_file)
        self.targets = get_last_code(data)
        dataset = dataset_encoding(data, icd9_diag_categories, max_visits)
        self.target_onehot = target_to_onehot(self.targets, dataset)
        intervals = np.moveaxis(dataset.copy(), 1, -1)[:, :, 0]
        target_torch = torch.from_numpy(self.target_onehot)
        dataset_torch = torch.from_numpy(dataset)
        intervals_torch = torch.from_numpy(intervals)
        self.data_tensor = TensorDataset(dataset_torch.transpose(1, -1), intervals_torch, target_torch)

     def __len__(self):
        return len(self.targets)

     def get_data(self):
        return self.data_tensor