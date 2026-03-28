from dataloader import get_dataloaders


train_loader , test_loader = get_dataloaders()

for signals , labels in train_loader:

    # 64 batches and every signal has 187 points
    print("Signals shape: " , signals.shape) # Signals shape:  torch.Size([64, 187])
    print("Labels shape: " , labels.shape) # Labels shape:  torch.Size([64])

    break