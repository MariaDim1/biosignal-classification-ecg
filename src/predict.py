import torch
import matplotlib.pyplot as plt

from dataloader import get_dataloaders
from model import ECGCNN


def show_predictions():

    _, test_loader = get_dataloaders()

    model = ECGCNN()
    model.load_state_dict(torch.load("ecg_model.pth"))
    model.eval()

    signals, labels = next(iter(test_loader))

    outputs = model(signals)
    _, preds = torch.max(outputs, 1)

    plt.figure(figsize=(10,5))

    for i in range(6):

        plt.subplot(2,3,i+1)
        plt.plot(signals[i])
        plt.title(f"P: {preds[i]} | T: {labels[i]}")
        plt.tight_layout()

    plt.savefig("results/plots/ecg_predictions.png")
    plt.show()


if __name__ == "__main__":
    show_predictions()