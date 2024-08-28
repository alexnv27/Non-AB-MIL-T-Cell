from torch.utils.data import DataLoader
import torch
from torch import nn
import MIL_DataLoader
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

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

def save_checkpoint(state, filename="", train_losses=None, test_losses=None, epoch=None):
    print("=> Saving checkpoint")
 
    # Check if the file already exists
    if os.path.exists(filename):
        # Load existing checkpoint
        checkpoint = torch.load(filename)
        
        # Append new data to existing lists
        checkpoint['epoch'].append(epoch)
        checkpoint['train_losses'].append(train_losses)
        checkpoint['test_losses'].append(test_losses)
        
        # Update state_dict and optimizer
        checkpoint['state_dict'] = state['state_dict']
        checkpoint['optimizer'] = state['optimizer']
        
        # Save updated checkpoint
        torch.save(checkpoint, filename)
    else:
        # Create new checkpoint
        checkpoint = {
            'epoch': [state['epoch']],  # Initialize as a list
            'state_dict': state['state_dict'],
            'optimizer': state['optimizer'],
            'train_losses': [train_losses],  # Initialize as a list
            'test_losses': [test_losses]  # Initialize as a list
        }
        
        # Save checkpoint
        torch.save(checkpoint, filename)

class NeuralNetwork(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(1024, 1)  

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = torch.tanh(x)  
        return x

model = NeuralNetwork(dropout_rate=0.5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
criterion = nn.SmoothL1Loss()

n_epochs = 50
patience = 3
best_loss = float('inf')
early_stop_counter = 0  
best_state = None


for epoch in range(n_epochs):
    running_loss_train = 0.0
    EachEightBatchLoss_train = []
    model.train()

    for patient_index, slide in enumerate(MIL_DataLoader.train_dataloader_retccl):
        patient_features = slide['features'].to(device)
        patient_Target = slide['t_cells'].to(device).to(torch.float32)

        tile_predictions = model(patient_features)
        pooled_predictions = custom_pooling(tile_predictions, mode='mean', top_k=1)

        loss = criterion(pooled_predictions, patient_Target[0])
        loss.backward()
       
        
        running_loss_train += loss.item()

        if (patient_index+1) % 8 == 0:
            optimizer.step()
            optimizer.zero_grad()
            print(f'TrainEpoch {epoch+1}, Batch {patient_index+1}, BatchLossSum: {running_loss_train/8:.4f}')
            EachEightBatchLoss_train.append(running_loss_train/8)
            running_loss_train = 0.0
    if (patient_index+1) % 8 != 0:
        optimizer.step()
        optimizer.zero_grad()  # Zero gradients after each update
        print(f'TrainEpoch {epoch+1}, Batch {patient_index+1}, BatchLossSum: {running_loss_train/(patient_index+1) % 8 :.4f}')
        EachEightBatchLoss_train.append(running_loss_train/(patient_index+1) % 8)
        running_loss_train = 0.0  # Reset running loss after each update

    epochAverageLossTrain = sum(EachEightBatchLoss_train) / len(EachEightBatchLoss_train)
    print(f'TrainEpoch {epoch+1}, AverageLossPerPatient: {epochAverageLossTrain:.4f}')
    writer.add_scalar("Loss/training", epochAverageLossTrain, epoch+1)   

    # Evaluate on test set
    model.eval()
    running_loss_test = 0.0
    EachEightBatchLoss_test = []

    with torch.no_grad():
        for patient_index, slide in enumerate(MIL_DataLoader.test_dataloader_retccl):    
            patient_features = slide['features'].to(device)
            patient_Target = slide['t_cells'].to(device).to(torch.float32)

            tile_predictions = model(patient_features)
            pooled_predictions = custom_pooling(tile_predictions, mode='mean', top_k=1)

            loss = criterion(pooled_predictions, patient_Target[0])
            running_loss_test += loss.item()

            if (patient_index+1) % 8 == 0:
                print(f'TestEpoch {epoch+1}, Batch {patient_index+1}, BatchLossSum: {running_loss_test:.4f}')
                EachEightBatchLoss_test.append(running_loss_test/8)
                running_loss_test = 0.0
        if (patient_index+1) % 8 != 0:
            print(f'TestEpoch {epoch+1}, Batch {patient_index+1}, BatchLossSum: {running_loss_test/(patient_index+1) % 8 :.4f}')
            EachEightBatchLoss_test.append(running_loss_test/(patient_index+1) % 8)
            running_loss_test = 0.0  # Reset running loss after each update

    epochAverageLossTest = sum(EachEightBatchLoss_test) / len(EachEightBatchLoss_test)
    print(f'TestEpoch {epoch+1}, AverageLossPerPatient: {epochAverageLossTest:.4f}')
    writer.add_scalar("Loss/testing", epochAverageLossTest, epoch+1)  

    # Save train and test losses each epoch
    checkpoint = {
        'epoch': [],
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_losses': [],
        'test_losses': []
    }
    save_checkpoint(checkpoint, filename="RMaxGraphCheckPoint.pth.tar", train_losses=epochAverageLossTrain,test_losses=epochAverageLossTest, epoch=(epoch+1))

    # Save best model based on test loss
    if epochAverageLossTest < best_loss:
        best_loss = epochAverageLossTest
        best_state = checkpoint
        early_stop_counter = 0
        save_checkpoint(best_state, filename="RMaxGraphBest.pth.tar", train_losses=epochAverageLossTrain, test_losses=epochAverageLossTest)

    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping")
            over=True
            break

writer.flush()
writer.close()
