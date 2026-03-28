import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import get_dataloaders
from model import ECGCNN


def train():

    train_loader , test_loader = get_dataloaders()

    model = ECGCNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters() , )

    epochs = 15

    for epoch in range(epochs):

        total_loss = 0

        for signals , labels in train_loader:

            optimizer.zero_grad()

            outputs = model(signals)

            loss = criterion(outputs , labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch + 1} , Loss: {avg_loss:.4f}")

    torch.save(model.state_dict() , "ecg_model.pth")
    print("Model saved!")

if __name__ == "__main__":
    train()