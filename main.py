import torch
from simpson_dataset import SimpsonDataset
from train_val_separate import train_val_get
from torch.utils.data import DataLoader
from model import Model, train

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(device)

    train_files, validation_files = train_val_get()

    train_dataset = SimpsonDataset(train_files, state="train")
    val_dataset = SimpsonDataset(validation_files, state="val")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = Model().to(device)

    train(model, device, train_loader, val_loader, 5, 64)


main()
