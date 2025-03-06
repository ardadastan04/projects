import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Define the custom dataset class
class CustomTextDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file, usecols=["explanation", "question", "answer", "label"])
        self.text = []

        for index, row in self.data.iterrows():
            self.text.append(f"{row['explanation']} {row['question']} {row['answer']} this is {row['label']}")

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.text[idx]

        if self.transform:
            sample = self.transform(sample)

        return {"text": sample}

# Specify the path to your CSV file
csv_file_path = 'train.csv'

# Create an instance of your custom dataset
dataset = CustomTextDataset(csv_file_path)

# Set batch size
batch_size = 64

# Create a data loader
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for batch in data_loader:
    print("Batch Content:", batch)
    break
    # Add more checks as needed