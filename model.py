from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.optim as optim

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
        for image, label in val_loader:
            with torch.no_grad():
                image = image.to(device)
                label = label.to(device)
                output = model(image)
                loss = criterion(output, label)
                writer.add_scalar("Loss/validation", loss, iteration_validation)
                iteration_validation+= 1

        print(f'Number of epoch: {epoch}')

    writer.close()
    torch.save(model.state_dict(), path_to_save_model)