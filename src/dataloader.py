from torch.utils.data import DataLoader 
from dataset import ECGDataset


def get_dataloaders():

    train_dataset = ECGDataset("data/raw/mitbih_train.csv")
    test_dataset = ECGDataset("data/raw/mitbih_test.csv")

    train_loader = DataLoader( # splits dataset into batches
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    return train_loader , test_loader