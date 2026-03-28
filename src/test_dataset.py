from dataset import ECGDataset


train_dataset = ECGDataset("data/raw/mitbih_train.csv")

print("Dataset size: " , len(train_dataset)) # Dataset size:  87554 (heartbeats)

signal , label = train_dataset[0]

print("Signal shape: " , signal.shape) # Signal shape:  torch.Size([187]) (187 points)
print("Label: " , label) # Label:  tensor(0) (category of ECG)
