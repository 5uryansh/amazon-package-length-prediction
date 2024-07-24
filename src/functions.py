import torch
import torch.nn as nn
# import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
# from tqdm import tqdm

# Define a custom dataset class that handles sparse data
class SparseCustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = csr_matrix(X)  # Store as a sparse matrix
        self.y = y.values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X_sample = torch.FloatTensor(self.X[idx].toarray()).squeeze()
        y_sample = torch.tensor(self.y[idx], dtype=torch.float32)
        return X_sample, y_sample
    
class NonLinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(NonLinearRegressionModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 4096)
        self.layer2 = nn.Linear(4096, 2048)
        self.layer3 = nn.Linear(2048, 128)
        self.layer4 = nn.Linear(128, 64)
        self.layer5 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.layer5(x)
        return x