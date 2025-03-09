import csv

import sklearn.metrics as metrics
import torch
from torch.utils.data import DataLoader

from model import Model, train
from simpson_dataset import SimpsonDataset
from train_val_separate import train_val_get
from visualise_results import visualise_results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_files, validation_files = train_val_get()

    train_dataset = SimpsonDataset(train_files, state="train")
    val_dataset = SimpsonDataset(validation_files, state="val")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = Model().to(device)

    train(model, device, train_loader, val_loader, "last", 30)


def test():
    device = "cuda"
    model = Model().to(device)
    model.load_state_dict(torch.load("last.pth"))
    test_files = []
    test_path = "simpson_test_set.csv"
    with open(test_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            test_files.append(line)

    test_dataset = SimpsonDataset(test_files, state="test")

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model.eval()

    y_true, y_pred, images = [], [], []

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)

            output = model(image)

            y_true.extend(label.cpu().numpy())
            y_pred.extend(output.cpu().numpy().argmax(-1))
            images.append(image)

    target_names = []
    for number_of_class in range(42):
        target_names.append(str(number_of_class))

    print(metrics.classification_report(y_true, y_pred, target_names=target_names))
    visualise_results(images, y_true, y_pred)


# main()
test()
