import torch.nn as nn


class ECGCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv1d(1 , 16 , kernel_size=3 , padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16 , 32 , kernel_size=3 , padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.fc = nn.Sequential(

            nn.Flatten(),
            nn.Linear(32 * 46 , 128),
            nn.ReLU(),
            nn.Linear(128 , 5) # 5 classes
        )

    def forward(self , x):

        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.fc(x)

        return x