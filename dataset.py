import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
# this is mine

def getUNSWData(batch_size, typedata="both", test_size=10000):
    # Load the data
    data_processed = pd.read_parquet("data/cic_processed.parquet")

    # Step 2: Define a custom Dataset
    class CSVDataSet(Dataset):
        def __init__(self, dataframe, input_features, target):
            self.data = dataframe
            self.inputs = self.data[input_features].values.astype(np.float32)
            self.targets = self.data[target].values.astype(np.int64)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            x = torch.tensor(self.inputs[index], dtype=torch.float32)
            y = torch.tensor(self.targets[index], dtype=torch.long)
            return x, y

    # Assuming 'label' is the target
    input_features = data_processed.columns.drop("Label")
    target = "Label"

    # Split the dataset
    train_data, test_data = train_test_split(
        data_processed, test_size=test_size, stratify=data_processed[target], random_state=42
    )

    # Balance the training dataset
    class_0_train = train_data[train_data["Label"] == 0]
    class_1_train = train_data[train_data["Label"] == 1]
    class_1_train_oversampled = class_1_train.sample(len(class_0_train), replace=True)
    balanced_train_data = pd.concat([class_0_train, class_1_train_oversampled])

    # Balance the testing dataset similarly
    class_0_test = test_data[test_data["Label"] == 0]
    class_1_test = test_data[test_data["Label"] == 1]
    class_1_test_oversampled = class_1_test.sample(len(class_0_test), replace=True)
    balanced_test_data = pd.concat([class_0_test, class_1_test_oversampled])

    # Shuffle the datasets
    balanced_train_data = balanced_train_data.sample(frac=1).reset_index(drop=True)
    balanced_test_data = balanced_test_data.sample(frac=1).reset_index(drop=True)

    # Save the balanced datasets as CSV files
    balanced_train_data.to_csv('cic_balanced_train_data.csv', index=False)
    balanced_test_data.to_csv('cic_balanced_test_data.csv', index=False)


    # Create the balanced train and test sets
    trainset = CSVDataSet(balanced_train_data, input_features, target)
    testset = CSVDataSet(balanced_test_data, input_features, target)


    # Create the DataLoader
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Print the balanced dataset class distributions
    print(balanced_train_data['Label'].value_counts())
    print(balanced_test_data['Label'].value_counts())

    if typedata == "both":
        return train_dataloader, test_dataloader
    if typedata == "train":
        return train_dataloader
    if typedata == "test":
        return test_dataloader

    # Print the balanced dataset class distributions
    # print(train_dataloader['Label'].value_counts())
    # print(test_dataloader['Label'].value_counts())

    if typedata == "both":
        return train_dataloader, test_dataloader
    if typedata == "train":
        return train_dataloader
    if typedata == "test":
        return test_dataloader


def getData(datasetname, batch_size=64, typedata="both", test_size=20000):

    if datasetname == "UNSW":
        getdataset = getUNSWData
    else:
        raise ValueError("Dataset not found")
    return getdataset(batch_size, typedata, test_size)
