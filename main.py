import numpy as np
import os
import torch
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch import nn

from data_loader import MedDataset
from model import BiteNet

admin_file = 'ADMISSIONS.csv'
diag_file = 'DIAGNOSES_ICD.csv'
procedure_file = 'PROCEDURES_ICD.csv'
prescript_file = 'PRESCRIPTIONS.csv'
drug_file = 'DRGCODES.csv'

datasets_path = '.\\mimic3_demo_data\\'

num_epochs = 10
lr = .001
embedding_dim = 128
n_heads = 4
output_dim = 19
n_codes = 19
n_visits = 5
blocks = 1
max_visits = 5

def main():
    dataset = MedDataset(max_visits, datasets_path, admin_file, diag_file, procedure_file, prescript_file, drug_file)
    med_dataset = dataset.get_data()
    dataset_size = len(med_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(.25 * dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_dataloader = DataLoader(med_dataset, batch_size=10,  sampler=train_sampler)
    val_dataloader = DataLoader(med_dataset, batch_size=10, sampler=val_sampler)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = BiteNet(embedding_dim, output_dim, n_heads, blocks, n_visits, n_codes).to(device)
    opt = Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_model = 1000

    for epoch in range(num_epochs):
        print("### Epoch: " + str(epoch) + " ###")
        # Perform Trainstep
        for i, (input, intervals, target) in enumerate(tqdm(train_dataloader)):
            model.train()
            input, intervals, target = input.to(device).long(), intervals.to(device).long(), target.to(device).float()
            opt.zero_grad()
            output = model(input, intervals)
            loss = criterion(output, target)
            print("\nTrain Loss: ", loss.detach().cpu())
            loss.backward()
            opt.step()

        for i, (input, intervals, target) in enumerate(tqdm(val_dataloader)):
            model.eval()
            input, intervals, target = input.to(device).long(), intervals.to(device).long(), target.to(device).float()
            output = model(input, intervals)
            val_loss = criterion(output, target)
            print("\nVal Loss: ", loss.detach().cpu())
            if val_loss.detach().cpu() < best_model:
                torch.save(model.state_dict(), "best_model_e_" + str(epoch) + "_loss_" + str(val_loss.detach().cpu()) + ".ckpt")
                best_model = val_loss.detach().cpu()

if __name__ == '__main__':
    main()