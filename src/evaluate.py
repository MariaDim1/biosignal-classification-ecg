import torch

from dataloader import get_dataloaders
from model import ECGCNN


def evaluate():

    _ , test_loader = get_dataloaders()

    model = ECGCNN()

    model.load_state_dict(torch.load("ecg_model.pth"))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for signals , labels in test_loader:

            outputs = model(signals)

            _ , predicted = torch.max(outputs , 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"Test Accuracy: {accuracy:.2f}%") # Test Accuracy: 98.12%

if __name__ == "__main__":
    evaluate()