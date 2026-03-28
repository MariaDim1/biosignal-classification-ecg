import pandas as pd
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):

    def __init__(self , file_path):
        
        data = pd.read_csv(file_path , header=None)

        self.X = data.iloc[: , :-1].values
        self.y = data.iloc[: , -1].values

    def __len__(self):

        return len(self.X)
    
    def __getitem__(self, idx):

        signal = torch.tensor(self.X[idx] , dtype=torch.float32)
        label = torch.tensor(self.y[idx] , dtype=torch.long)

        return signal , label