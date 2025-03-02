import torch
from simpson_dataset import SimpsonDataset
from train_val_separate import train_val_get
from torch.utils.data import DataLoader
from model import Model, train
import csv
import sklearn.metrics as metrics

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_files, validation_files = train_val_get()

    train_dataset = SimpsonDataset(train_files, state="train")
    val_dataset = SimpsonDataset(validation_files, state="val")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = Model().to(device)
    model.load_state_dict(torch.load('iteration1.pth'))

    train(model, device, train_loader, val_loader, "iteration2.pth",  5, 64)


def test():
    device = 'cuda'
    model = Model().to(device)
    model.load_state_dict(torch.load('iteration2.pth'))
    test_files = []
    test_path = "simpson_test_set.csv"
    with open(test_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            test_files.append(line)

    test_dataset = SimpsonDataset(test_files, state="test")

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)

            output = model(image)

            y_true.extend(label.cpu().numpy())
            y_pred.extend(output.cpu().numpy().argmax(-1))

    f1_macro = metrics.f1_score(y_true, y_pred, average='macro')  # Среднее по классам
    f1_micro = metrics.f1_score(y_true, y_pred,
                        average='micro')  # Учитывает общее количество истинных положительных, ложных положительных и ложных отрицательных
    f1_weighted = metrics.f1_score(y_true, y_pred, average='weighted')  # Учитывает поддержку каждого класса

    print(f'F1 Macro: {f1_macro}')
    print(f'F1 Micro: {f1_micro}')
    print(f'F1 Weighted: {f1_weighted}')


main()
test()