import matplotlib.pyplot as plt
import numpy as np
import torch


def visualise_results(images, y_true, y_pred):
    plt.figure(figsize=(12, 12))

    mean = torch.tensor([0.4621, 0.4078, 0.3531]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.2559, 0.2337, 0.2644]).view(1, 3, 1, 1).cuda()

    images = [img * std + mean for img in images]

    images_to_visualize = []
    for index_of_batch in range(0, 24, 2):
        image = images[index_of_batch][0]
        image = image.permute(1, 2, 0)
        image = image.cpu().numpy()
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
        images_to_visualize.append(image)

    for index in range(12):
        plt.subplot(4, 3, index + 1)
        plt.imshow(images_to_visualize[index])
        plt.title(f"True: {y_true[index * 32 * 2]}, Pred: {y_pred[index * 32 * 2]}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("pictures/test.png")
