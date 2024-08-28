from torch.utils.data import DataLoader
import torch
from torch import nn
import MIL_DataLoader
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

class SunnyResectionNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(SunnyResectionNN, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(1024, 1)  
        self

    
    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        print(f"Shape after fc1: {x.shape}")
        x = self.fc2(x)
        x = torch.tanh(x)  
        print(f"Output shape: {x.shape}")
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.01)

def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
load_model=True

if load_model:
    load_checkpoint(torch.load("/home/air/Alex/ImportantModelCheckpoints/ActualBestRetCCLb.pth.tar"))
    
n_epochs = 1
data = { 'patient_ID': [], 'Predictions': []}
model.eval()  # Set the model to evaluation mode

for epoch in range(n_epochs):
    checkpoint={'state_dict':model.state_dict(), 'optimizer': optimizer.state_dict()}

    with torch.no_grad():  # Disable gradient calculation for testing
        for patient_index, slide in enumerate(MIL_DataLoader.sunyResection_dataloader_retccl):
            #print(patient_index+1)
            #print(slide)
            patient_features = (slide['features']).to(device)
            patient_ID = (slide['ID'])
    
            tile_predictions = model(patient_features)
            print(tile_predictions)
            pooled_predictions=custom_pooling(tile_predictions, mode='mean', top_k=5)#print(custom_pooling(tile_predictions, mode='mean_top_k', top_k=1 ))
            data['Predictions'].append(pooled_predictions.item())  # Convert tensor to Python float
            data['patient_ID'].append(patient_ID[0])

df_predictions = pd.DataFrame(data)
df_predictions.to_excel('TTest.xlsx', index=False)
print("done")
#print(df_predictions)   



