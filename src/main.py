import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
# from scipy.sparse import csr_matrix
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

from functions import SparseCustomDataset
from functions import NonLinearRegressionModel

import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"



# parsing
parser = argparse.ArgumentParser(description="Package Length Predictor")
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--max_grad_norm', type=float, default=1.0)

args = parser.parse_args()
lr = args.learning_rate
epochs = args.epochs
batch_size = args.batch_size
max_grad_norm = args.max_grad_norm



# Loading the dataset
train = pd.read_csv('../dataset/train_final.csv')
train = train.dropna()
print("\n\nDataset loaded successfully!!!")



# Text preprocessing and splitting the dataset
print("\n\nInitiating data preprocessing, this can take upto 10 minutes...")
train['FINAL_STRING'] = train['FINAL_STRING'].str.lower()
X_train, X_test, y_train, y_test = train_test_split(train['FINAL_STRING'], train['PRODUCT_LENGTH'], test_size=0.2, random_state=42)

# Removing outliers
Q1 = y_train.quantile(0.25)
Q3 = y_train.quantile(0.75)
IQR = Q3 - Q1
outliers = y_train[(y_train < (Q1 - 1.5 * IQR)) | (y_train > (Q3 + 1.5 * IQR))]

y_train_clean = y_train.drop(outliers.index)
X_train_clean = X_train.loc[y_train_clean.index]
X_train = X_train_clean
y_train = y_train_clean

# Convert text data into TF-IDF features
print("Vectorising the dataset....")
vectorizer = TfidfVectorizer(max_features=50000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("Vectorisation completed!!!")

# Create datasets and dataloaders
train_dataset = SparseCustomDataset(X_train_tfidf, y_train)
test_dataset = SparseCustomDataset(X_test_tfidf, y_test)

batch_size = batch_size

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("Data preprocessing completed!!!")



# model and training
input_dim = X_train_tfidf.shape[1]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n\nCurrently using {device}....")
model = NonLinearRegressionModel(input_dim)
# Check if the file exists
if os.path.isfile('model_state_dict.pth'):
    # Load the model state dict if the file exists
    model.load_state_dict(torch.load('model_state_dict.pth'))
    print("Model state loaded successfully.")
else:
    # Save the model state dict if the file does not exist
    torch.save(model.state_dict(), 'model_state_dict.pth')
    print("Model state saved successfully.")

# Move the model to the device
model = model.to(device)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# dynamic learning rate
accumulation_steps = 2
effective_batch_size = batch_size * accumulation_steps

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
gamma = 0.1
# Gradient clipping
max_grad_norm = max_grad_norm

# Training loop
print(f"Learning Rate: {lr}\nBatch Size: {batch_size}\nEpochs: {epochs}\nGrad Norm: {max_grad_norm}")
print("\n\n\nTraining....")
num_epochs = epochs
for epoch in tqdm(range(num_epochs), desc="Epochs"):
    model.train()
    epoch_loss = 0.0  # Track loss for the entire epoch
    optimizer.zero_grad()
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)

        # Backward pass
        loss.backward()

        # Clip gradients preventing from drastic change in gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Accumulate gradients
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            # lr = lr*gamma
            # print(f"\nLearning rate changed to {lr}")

        # Accumulate loss
        epoch_loss += loss.item()

        # Print batch loss for every 100th batch
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'model_state_dict.pth')
    # Print epoch loss
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}')
    
# Define your model
model = NonLinearRegressionModel(input_dim)

torch.save(model.state_dict(), 'model_state_dict.pth')    
print("Model saved.")
    
print("Testing....")


# Evaluation
model.eval()
losses = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = torch.tensor(X_batch, dtype=torch.float32).to(device)
        y_batch = torch.tensor(y_batch, dtype=torch.float32).to(device)
        outputs = model(X_batch).to(device)
        loss = criterion(outputs.squeeze(), y_batch)
        losses.append(loss.item())

losses = np.array(losses)
max_loss = np.max(losses)
min_loss = np.min(losses)
avg_loss = np.mean(losses)
loss_variance = np.var(losses)
loss_std_dev = np.std(losses)

print(f"Maximum Loss: {max_loss:.4f}")
print(f"Minimum Loss: {min_loss:.4f}")
print(f"Average Loss: {avg_loss:.4f}")
print(f"Loss Variance: {loss_variance:.4f}")
print(f"Loss Standard Deviation: {loss_std_dev:.4f}")