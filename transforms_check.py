import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from train_val_separate import train_val_get


def read_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class SimpsonDataset(Dataset):
    def __init__(self, files, state="train"):
        self.img_labels = files
        self.state = state
        if state == "train":
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(size=(224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=(-30, 30)),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4621, 0.4078, 0.3531], [0.2559, 0.2337, 0.2644]
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4621, 0.4078, 0.3531], [0.2559, 0.2337, 0.2644]
                    ),
                ]
            )

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels[idx][1]
        image = read_image(img_path)
        label = int(self.img_labels[idx][0])
        image = self.transform(image)
        return image, label, img_path


def main():
    train_files, validation_files = train_val_get()

    train_dataset = SimpsonDataset(train_files, state="train")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    for image, label, path in train_loader:
        image = image.squeeze()
        image = image.permute(1, 2, 0)
        image = image.numpy()
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        orig = cv2.imread(path[0])
        orig = cv2.resize(orig, (image.shape[1], image.shape[0]))
        image = cv2.hconcat([image, orig])
        cv2.imshow("a", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


main()
