from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics as metrics
from torch.utils.tensorboard import SummaryWriter

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.CNN = models.resnet50(weights='IMAGENET1K_V1')
        num_features_on_fc_input = self.CNN.fc.in_features
        self.CNN.fc = torch.nn.Linear(num_features_on_fc_input, 42)

    def forward(self, img1):
        return self.CNN(img1)


def train(model, device, train_loader, val_loader, path_to_save_model, epochs=50, batch_size=64):
    writer = SummaryWriter("runs/Simpsons")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    iteration_train = 0
    iteration_validation = 0
    for epoch in range(epochs):
        model.train()
        for image, label in train_loader:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            writer.add_scalar("Loss/train", loss, iteration_train)
            iteration_train+=1
            loss.backward()
            optimizer.step()

        model.eval()
        y_true = []
        y_pred = []
        for image, label in val_loader:
            with torch.no_grad():
                image = image.to(device)
                label = label.to(device)
                output = model(image)
                y_true.extend(label.cpu().numpy())
                y_pred.extend(output.cpu().numpy().argmax(-1))
                iteration_validation+= 1

        target_names = []
        for number_of_class in range(42):
            target_names.append(str(number_of_class))

        result = metrics.classification_report(y_true, y_pred, target_names=target_names)
        result = result.splitlines()

        for line_index in range(2, 43):
            # precision - recall - f1-score
            current_class_results = result[line_index].split()
            current_class_results = [float(current_class_results[elem_index]) for elem_index in range(1, 4)]
            writer.add_scalar(f"Precision of {line_index} class", current_class_results[0], epoch)
            writer.add_scalar(f"Recall of {line_index} class", current_class_results[1], epoch)
            writer.add_scalar(f"f1-score of {line_index} class", current_class_results[2], epoch)


        print(f'Number of epoch: {epoch}')

    writer.close()
    torch.save(model.state_dict(), path_to_save_model)