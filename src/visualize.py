import matplotlib.pyplot as plt
from dataset import ECGDataset
import os


dataset = ECGDataset("data/raw/mitbih_train.csv")

for i in range(len(dataset)):

    signal, label = dataset[i]

    if label != 0:
        print("Found label:", label, "at index:", i)
        break

signal , label = dataset[72471]

plt.plot(signal)
plt.title(f"ECG Signal - Label: {label}")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.savefig(f"results/plots/ecg_sample_{label}.png")
plt.show()