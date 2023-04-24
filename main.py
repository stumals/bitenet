import numpy as np
import os
import torch
from torch.optim import Adam
import pandas as pd
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch import nn
from torchmetrics import Accuracy, Precision, Recall, JaccardIndex
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryJaccardIndex

#from data_loader import MedDataset
from model import BiteNet

from dataset import data_prep, MedDataset

"""admin_file = 'ADMISSIONS.csv'
diag_file = 'DIAGNOSES_ICD.csv'
procedure_file = 'PROCEDURES_ICD.csv'
prescript_file = 'PRESCRIPTIONS.csv'
drug_file = 'DRGCODES.csv'
datasets_path = 'mimic3_demo_data'"""

admin_file = 'admissions_full.csv'
diag_file = 'diagnoses_icd_full.csv'
procedure_file = 'procedures_full.csv'
datasets_path = 'dataset_full'

num_epochs = 20
lr = .001
embedding_dim = 128
n_heads = 4
output_dim = 19
n_codes = 19
n_visits = 10
blocks = 1
max_visits = 10
batch_size = 32
checkpoint_path = "checkpoints"
log_path = "logs"
log_name = "binary_class_metrics_pos.csv"

"""def log_metrics(metrics, out_path, filename):
    for metric in metrics:"""


class BinaryMetrics:
    def __init__(self):
        self.accuracy = BinaryAccuracy(task="multiclass", num_classes=19)
        self.precision_5 = BinaryPrecision(task="multiclass", top_k=5, num_classes=19)
        self.precision_20 = BinaryPrecision(task="multiclass", top_k=20, num_classes=19)
        self.recall = BinaryRecall(task="multiclass", num_classes=19)
        self.epoch_metrics = {"train_loss": [], "train_acc": [], "train_pred5": [], "train_pred20": [], "train_recall": [],
                              "val_loss": [], "val_acc": [], "val_pred5": [], "val_pred20": [], "val_recall": []}

    def log_metrics(self, loss, metrics, output, target, train=True):
        if train:
            metrics["train_loss"] += loss.detach().cpu().item()
            metrics["train_acc"] += self.accuracy(output, target).item()
            metrics["train_pred5"] += self.precision_5(output, target).item()
            metrics["train_pred20"] += self.precision_20(output, target).item()
            metrics["train_recall"] += self.recall(output, target).item()
            metrics["train_iter"] += 1
        else:
            metrics["val_loss"] += loss.detach().cpu().item()
            metrics["val_acc"] += self.accuracy(output, target).item()
            metrics["val_pred5"] += self.precision_5(output, target).item()
            metrics["val_pred20"] += self.precision_20(output, target).item()
            metrics["val_recall"] += self.recall(output, target).item()
            metrics["val_iter"] += 1
        return metrics

    def print_metrics(self, metrics):
        train_iter = metrics["train_iter"]
        val_iter = metrics["val_iter"]
        print("\nTrain Loss: ", metrics["train_loss"] / train_iter)
        print("Train acc: ", metrics["train_acc"] / train_iter)
        print("Train prec@5: ", metrics["train_pred5"] / train_iter)
        print("Train recall: ", metrics["train_recall"] / train_iter)

        print("\nVal Loss: ", metrics["val_loss"] / val_iter)
        print("Val acc: ", metrics["val_acc"] / val_iter)
        print("Val prec@5: ", metrics["val_pred5"] / val_iter)
        print("Val recall: ", metrics["val_recall"] / val_iter)
        self.log_epoch(metrics)

    def log_epoch(self, metrics):
        train_iter = metrics["train_iter"]
        val_iter = metrics["val_iter"]
        self.epoch_metrics["train_loss"].append(metrics["train_loss"] / train_iter)
        self.epoch_metrics["train_acc"].append(metrics["train_acc"] / train_iter)
        self.epoch_metrics["train_pred5"].append(metrics["train_pred5"] / train_iter)
        self.epoch_metrics["train_pred20"].append(metrics["train_pred20"] / train_iter)
        self.epoch_metrics["train_recall"].append(metrics["train_recall"] / train_iter)

        self.epoch_metrics["val_loss"].append(metrics["val_loss"] / val_iter)
        self.epoch_metrics["val_acc"].append(metrics["val_acc"] / val_iter)
        self.epoch_metrics["val_pred5"].append(metrics["val_pred5"] / val_iter)
        self.epoch_metrics["val_pred20"].append(metrics["val_pred20"] / val_iter)
        self.epoch_metrics["val_recall"].append(metrics["val_recall"] / val_iter)

    def write_metrics(self, outpath, filename):
        data = pd.DataFrame.from_dict(self.epoch_metrics, orient="columns")
        data.to_csv(os.path.join(outpath, filename))


def main():
    print("LOADING DATA")
    data_array, target_array = data_prep(datasets_path, admin_file, diag_file, procedure_file,
                                         min_visits=2, max_visits=max_visits)
    # dataset = MedDataset(max_visits, datasets_path, admin_file, diag_file, procedure_file, prescript_file, drug_file)
    dataset = MedDataset(data_array, target_array)
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

    train_dataloader = DataLoader(med_dataset, batch_size=batch_size,  sampler=train_sampler)
    val_dataloader = DataLoader(med_dataset, batch_size=batch_size, sampler=val_sampler)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Loading Model")
    model = BiteNet(embedding_dim, output_dim, n_heads, blocks, n_visits, n_codes).to(device)
    opt = Adam(model.parameters(), lr=lr)
    #opt = torch.optim.RMSprop(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    metric_class = BinaryMetrics()

    best_model = 1000
    print("Training")
    for epoch in range(num_epochs):
        metrics = {"train_loss": 0, "val_loss": 0, "train_iter": 0, "val_iter": 0, "train_acc": 0,
                   "train_pred5": 0, "train_pred20": 0, "train_recall": 0, "val_acc": 0, "val_pred5": 0, "val_pred20": 0, "val_recall": 0}

        print("\n### Epoch: " + str(epoch) + " ###")
        # Perform Trainstep
        for i, (input, intervals, target) in enumerate(tqdm(train_dataloader)):
            model.train()
            input, intervals, target = input.to(device).long(), intervals.to(device).long(), target.to(device).float()
            opt.zero_grad()
            output = model(input, intervals)
            loss = criterion(output, target)
            loss.backward()
            opt.step()
            metrics = metric_class.log_metrics(loss, metrics, output, target, train=True)

        for i, (input, intervals, target) in enumerate(tqdm(val_dataloader)):
            model.eval()
            input, intervals, target = input.to(device).long(), intervals.to(device).long(), target.to(device).float()
            output = model(input, intervals)
            loss = criterion(output, target)
            metrics = metric_class.log_metrics(loss, metrics, output, target, train=False)

        if metrics["val_loss"] / metrics["val_iter"] < best_model:
            torch.save(model.state_dict(), os.path.join(checkpoint_path, "sinusoidal_best_model_e_" + str(epoch) + "_loss_" + str(metrics["val_loss"] / metrics["val_iter"])[:6] + ".ckpt"))
            best_model = metrics["val_loss"] / metrics["val_iter"]

        metric_class.print_metrics(metrics)
    metric_class.write_metrics(log_path, log_name)


if __name__ == '__main__':
    main()
