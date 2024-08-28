import os
import h5py
import fileinput
import pandas as pd
import numpy as np
import glob
import torch 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
masterfile_path = '/mnt/synology/ICB_Data_SUNY/merged_masterfile_tme_signatures.csv'
df=pd.read_csv(masterfile_path)

class NewmilDataset(Dataset):

    def __init__(self, df, task_counts=None, bag_size=None):
        super(NewmilDataset, self).__init__()
        self.bag_size = bag_size
        self.df = [group for _, group in df.groupby('ID', sort=False)]
        
    def __getitem__(self, idx):
        df_row = self.df[idx]

        try:
            T_cells = df_row['T_cells'].iloc[0] 
          
        except Exception as e:
            #print(f"Error accessing T_cells: {e}")
            T_cells = None  # or you can handle it in another way

        patient_ID=df_row['ID'].iloc[0] 
        slide_path=df_row['file_path'].iloc[0] 
        features = []
        coords = []
        paths=[]
     
        if 0 <= idx < len(self.df):
            for file_path in self.df[idx]['file_path']:
                if os.path.isfile(file_path):
                        with h5py.File(file_path, 'r') as f: 
                            features.append(f['features'][()])
                            coords.append(f['coords'][()])
                            paths.append(file_path)
                else: 
                    print(f"File does not exist: {file_path}")
            ft_np = np.concatenate(features, 0)
            coords_np = np.concatenate(coords, 0)
            slide_path = paths
      
  
            #remove na values in TCGA-60-2710
            if np.isnan(ft_np).any():
                ft_np = ft_np[~np.isnan(ft_np).any(axis=1)] #not sure what this does, maybe removes na values

            ft_pt = torch.from_numpy(ft_np)
            if self.bag_size: #not sure what this does
                #ft_pt, ft_len = _to_fixed_size_bag(ft_pt, bag_size=self.bag_size) 
                print()
            else:
                ft_len = len(ft_pt)
            
            data = {}
            data['features'] = ft_pt
            assert not ft_pt.isnan().any(), slide_path #not sure what this does
            data['ft_lengths/tile#'] = torch.from_numpy(np.asarray(ft_len))
            data['slide_path'] = slide_path
            data['t_cells']=T_cells
            data['coordiantes']=coords_np
            data['ID']=patient_ID
            
            return data
        else:
            print(f"File does not exist: {slide_path}")
    def __len__(self):
        return len(self.df)

dataset=NewmilDataset(df)

#print(dataset.__getitem__(1))
#print(dataset.__getitem__(1502))
#print(len(dataset))
#print(len(dataset))  
   


