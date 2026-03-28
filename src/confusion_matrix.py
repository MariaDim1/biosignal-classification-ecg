import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataloader import get_dataloaders
from model import ECGCNN


def plot_confusion_matrix():

    _, test_loader = get_dataloaders()

    model = ECGCNN()
    model.load_state_dict(torch.load("ecg_model.pth"))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for signals, labels in test_loader:

            outputs = model(signals)

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.title("ECG Confusion Matrix")
    plt.savefig("results/plots/ecg_confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    plot_confusion_matrix()