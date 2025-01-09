import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler,Subset
from sklearn.preprocessing import MinMaxScaler 
def standardization(data):
    data = data
    mu = np.mean(data, axis=0)
    # print("mu:",mu,mu.shape)
    sigma = np.std(data, axis=0)
    # print("sigma:",sigma,sigma.shape)
    return (data - mu) / sigma

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range,_range,np.min(data)

class MyDataset(Dataset):
    def __init__(self, signal_file, label_file):
        self.data = np.load(signal_file).transpose(0, 2, 1)
        # self.data = normalization(self.data)
        self.label = np.load(label_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        return data, label


def fs_dataset(signal_file, label_file,samples_per_class = 10, num_classes =10):
    dataset = MyDataset(signal_file, label_file)

    batch_size = samples_per_class * num_classes
    subset_indices = []

    for label in range(num_classes):
        # Get all indices for the current label
        label_indices = [idx for idx, (_, lbl) in enumerate(dataset) if lbl == label]
        # Shuffle indices for each class
        np.random.shuffle(label_indices)
        # Select samples_per_class number of samples for each class
        selected_indices = label_indices[:samples_per_class]
        subset_indices.extend(selected_indices)

    # Shuffle the indices to mix classes
    np.random.shuffle(subset_indices)
    # print(subset_indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(subset_indices))
    data = next(iter(dataloader))
    signal, label = data
    # print(label)
    # signal, label = signal, label.numpy()
    remaining_indices = set(range(len(dataset))) - set(subset_indices)
    # Create a new DataLoader for validation using the remaining indices and with a batch size equal to the remaining samples
    remaining_dataset = Subset(dataset, list(remaining_indices))
    remaining_dataloader = DataLoader(remaining_dataset, batch_size=len(remaining_dataset), shuffle=True)
    val_data = next(iter(remaining_dataloader))
    val_signal, val_label = val_data
    # print(signal.shape,label.shape,val_signal.shape,val_label.shape)
    return signal.numpy(),label.numpy(),val_signal.numpy(),val_label.numpy()
data_path = 'Dataset_4800/X_train_30Class.npy'
label_path = 'Dataset_4800/Y_train_30Class.npy'
fs_dataset(data_path, label_path,samples_per_class = 1, num_classes =10)