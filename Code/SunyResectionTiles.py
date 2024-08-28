from torch.utils.data import DataLoader
import torch
from torch import nn
import MIL_DataLoader
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import os
import h5py
import shutil
'''
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

class SunnyResectionNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(SunnyResectionNN, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(1024, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        return x

def custom_pooling(tensor, mode='mean', top_k=None):
    if mode == 'mean':
        return torch.mean(tensor)
    elif mode == 'median':
        return torch.median(tensor)
    elif mode == 'max':
        return torch.max(tensor)
    elif mode == 'mean_top_k' and top_k is not None:
        values, _ = torch.topk(tensor.view(-1), top_k)
        return torch.mean(values)
    else:
        raise ValueError(f"Unsupported pooling mode: {mode}")

model = SunnyResectionNN(dropout_rate=0.5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

load_model = True
criterion = nn.SmoothL1Loss

if load_model:
    load_checkpoint(torch.load("RTop5allSlides.pth.tar"))

n_epochs = 1
model.eval()  # Set the model to evaluation mode

# Directory to save HDF5 files
output_dir = 'HDF5FilePredictions1'
os.makedirs(output_dir, exist_ok=True)
patient_files = {}

excel_data = {
    'patient_ID': [],
    'slide_index': [],
    'file_path': []
}

for epoch in range(n_epochs):
    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}

    with torch.no_grad():  # Disable gradient calculation for testing
        for patient_index, slide in enumerate(MIL_DataLoader.sunyResection_Tiles_dataloader_retccl):
        
            slide_path = slide['slide_path'][0]
            patient_features = slide['features'].to(device)
            patient_ID = slide['ID'][0]  # Assuming ID is a list and we need the first element

            tile_predictions = model(patient_features)
            pooled_predictions = custom_pooling(tile_predictions, mode='mean', top_k=5)
            
            
            # Create a dictionary to store the predictions
            slide_data = {
                'pooled_predictions': pooled_predictions.cpu().numpy(),
                'tile_predictions': tile_predictions.cpu().numpy()
            }

            if patient_ID not in patient_files:
                patient_files[patient_ID] = 0
            else:
                patient_files[patient_ID] += 1

            slide_index = patient_files[patient_ID]
             # Extract the file name from the slide path
            file_name = os.path.basename(slide_path)
            hdf5_path = os.path.join(output_dir, file_name)

            # Save the predictions to an HDF5 file
            with h5py.File(hdf5_path, 'w') as f:
                f.create_dataset('pooled_predictions', data=slide_data['pooled_predictions'])
                f.create_dataset('tile_predictions', data=slide_data['tile_predictions'])

            excel_data['patient_ID'].append(patient_ID)
            excel_data['slide_index'].append(slide_index)
            excel_data['file_path'].append(slide_path)

            shutil.copy(slide_path, output_dir)

            
df_excel = pd.DataFrame(excel_data)

# Sort the DataFrame by patient_ID and slide_index
df_excel.sort_values(by=['patient_ID', 'slide_index'], inplace=True)

# Save to Excel
#df_excel.to_excel('RetCCLTop5SunySlideTilepredictions_master.xlsx', index=False)

'''
# Ensure the folder exists
destination_folder = 'HDF5FilePredictions'

# Ensure the folder exists
if os.path.exists(destination_folder):
    # Get list of HDF5 files in the folder
    hdf5_files = [os.path.join(destination_folder, file) for file in os.listdir(destination_folder) if file.endswith('.hdf5')]

    # Print contents of each HDF5 file
    for file_path in hdf5_files:
        with h5py.File(file_path, 'r') as hf:
            print(f"Contents of HDF5 file: {file_path}")
            for key in hf.keys():
                if hf[key].shape == ():  # Handle scalar datasets
                    print(f"{key}: {hf[key][()]}")
                else:
                    print(f"{key}: {hf[key][:]}")
        print()  # Add a blank line between files for clarity
else:
    print(f"Folder '{destination_folder}' does not exist or is empty.")
   