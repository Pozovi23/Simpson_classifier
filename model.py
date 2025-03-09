import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import models


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.CNN = models.resnet50(weights="IMAGENET1K_V1")
        num_features_on_fc_input = self.CNN.fc.in_features
        self.CNN.fc = torch.nn.Linear(num_features_on_fc_input, 42)

    def forward(self, img1):
        return self.CNN(img1)


def train(model, device, train_loader, val_loader, path_to_save_model, epochs=50):
    writer = SummaryWriter("runs/Simpsons")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        print(f"Number of epoch: {epoch}")
        total_train_loss = 0
        num_of_train_batches = 0
        model.train()
        for image, label in train_loader:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            writer.add_scalar(
                "Loss/train each batch", loss.item(), num_of_train_batches
            )
            total_train_loss += loss.item()
            num_of_train_batches += 1
            loss.backward()
            optimizer.step()

        writer.add_scalar("Loss/train", total_train_loss / num_of_train_batches, epoch)

        model.eval()
        y_true, y_pred = [], []

        total_validation_loss = 0
        num_of_validation_batches = 0

        for image, label in val_loader:
            with torch.no_grad():
                image = image.to(device)
                label = label.to(device)
                output = model(image)
                loss = criterion(output, label)
                writer.add_scalar(
                    "Loss/validation each batch", loss.item(), num_of_validation_batches
                )
                total_validation_loss += loss.item()
                num_of_validation_batches += 1
                y_true.extend(label.cpu().numpy())
                y_pred.extend(output.cpu().numpy().argmax(-1))

        writer.add_scalar(
            "Loss/validation", total_validation_loss / num_of_validation_batches, epoch
        )

        target_names = []
        for number_of_class in range(42):
            target_names.append(str(number_of_class))

        result = metrics.classification_report(
            y_true, y_pred, target_names=target_names
        )
        result = result.splitlines()

        for line_index in range(2, 44):
            # precision - recall - f1-score
            current_class_results = result[line_index].split()
            current_class_results = [
                float(current_class_results[elem_index]) for elem_index in range(1, 4)
            ]
            writer.add_scalar(
                f"Precision of {line_index - 2} class", current_class_results[0], epoch
            )
            writer.add_scalar(
                f"Recall of {line_index - 2} class", current_class_results[1], epoch
            )
            writer.add_scalar(
                f"f1-score of {line_index - 2} class", current_class_results[2], epoch
            )

        if epoch == 9:
            torch.save(model.state_dict(), path_to_save_model + "_10_epochs.pth")

        if epoch == 14:
            torch.save(model.state_dict(), path_to_save_model + "_15_epochs.pth")

        if epoch == 19:
            torch.save(model.state_dict(), path_to_save_model + "_20_epochs.pth")

        if epoch == 24:
            torch.save(model.state_dict(), path_to_save_model + "_25_epochs.pth")

    writer.close()
    torch.save(model.state_dict(), path_to_save_model + ".pth")
