from torch.utils.data import DataLoader
import pandas as pd
import Newmil_Dataset
import glob
import os
import Newmil_DatasetSlide


def prepare_data(df,df2, model_type):
    if model_type == 'transpath':
        # No modification needed for transpath
        df_model = df.copy()
        test_df=df2.copy()
    elif model_type == 'UNI':
        df_model = df.copy()
        test_df=df2.copy()
        df_model['file_path'] = df_model['file_path'].str.replace('PrivateDataFilePathRedacted')
        test_df['file_path'] = test_df['file_path'].str.replace('PrivateDataFilePathRedacted', 'PrivateDataFilePathRedacted')
    elif model_type == 'retccl':
        df_model = df.copy()
        test_df=df2.copy()
        df_model['file_path'] = df_model['file_path'].str.replace('PrivateDataFilePathRedacted', 'PrivateDataFilePathRedacted')
        test_df['file_path'] = test_df['file_path'].str.replace('PrivateDataFilePathRedacted', 'PrivateDataFilePathRedacted')
    
    training_data = df_model[df_model['file_path'].str.contains('TCGA')]
    testing_data = df_model[df_model['file_path'].str.contains('C3N') | df_model['file_path'].str.contains('C3L')]
    PrivateDataNameRedacted_data = test_df

    training_dataset = Newmil_Dataset.NewmilDataset(training_data)
    training_datasetslides = Newmil_DatasetSlide.NewmilDataset(training_data)

    testing_dataset = Newmil_Dataset.NewmilDataset(testing_data)
    testing_datasetslides = Newmil_DatasetSlide.NewmilDataset(testing_data)


    PrivateDataNameRedacted_dataset = Newmil_Dataset.NewmilDataset(PrivateDataNameRedacted_data)
    PrivateDataNameRedacted_datasettiles = Newmil_DatasetSlide.NewmilDataset(PrivateDataNameRedacted_data)


    train_dataloader = DataLoader(training_dataset, batch_size=1, shuffle=True)
    train_dataloaderslides=DataLoader(training_datasetslides, batch_size=1, shuffle=True)

    test_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=True)
    test_dataloaderslides=DataLoader(testing_datasetslides, batch_size=1, shuffle=True)

    PrivateDataNameRedacted_dataloader = DataLoader(PrivateDataNameRedacted_dataset, batch_size=1, shuffle=True)
    PrivateDataNameRedactedTiles_dataloader = DataLoader(PrivateDataNameRedacted_datasettiles, batch_size=1, shuffle=True)

    return train_dataloader, test_dataloader, PrivateDataNameRedacted_dataloader, train_dataloaderslides, test_dataloaderslides, PrivateDataNameRedactedTiles_dataloader

masterfile_path = 'PrivateDataFilePathRedacted'
df = pd.read_csv(masterfile_path)

TEST_PATH = 'PrivateDataFilePathRedacted*.hdf5'
test_files = glob.glob(TEST_PATH)
df2 = pd.DataFrame({'ID': [os.path.basename(x)[:14] for x in test_files],
                            'file_path':test_files,
                            'T_cells':0})

# Transpath model
train_dataloader_transpath, test_dataloader_transpath, PrivateDataFilePathRedacted_dataloader_transpath,train_dataloader_slides_transpath, test_dataloader_slides_transpath, PrivateDataFilePathRedacted_Tiles_dataloader_transpath = prepare_data(df, df2, 'transpath')

# UNI model
train_dataloader_uni, test_dataloader_uni, PrivateDataFilePathRedacted_dataloader_uni,train_dataloader_slides_uni,test_dataloader_slides_uni, PrivateDataFilePathRedacted_Tiles_dataloader_uni = prepare_data(df, df2, 'UNI')

# Retccl model
train_dataloader_retccl, test_dataloader_retccl, PrivateDataFilePathRedacted_dataloader_retccl,train_dataloader_slides_retccl,test_dataloader_slides_retccl, PrivateDataFilePathRedacted_Tiles_dataloader_retccl = prepare_data(df, df2, 'retccl')

