import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# === Custom Dataset ===
class CSVDataset(Dataset):
    def __init__(self, filepath, scaler=None, fit=False):
        df = pd.read_csv(filepath)
        label_col = df.columns[-1]
        df[label_col] = df[label_col].str.upper().map({'NO': 0, 'YES': 1})
        X = df.iloc[:, :-1].values
        y = df[label_col].values

        if fit:
            self.scaler = StandardScaler().fit(X)
            joblib.dump(self.scaler, "scaler.pkl")
        else:
            self.scaler = joblib.load("scaler.pkl")
        
        X_scaled = self.scaler.transform(X)
        self.X = torch.tensor(X_scaled, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === Multi-Head Model ===
class MultiHeadHospitalNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, 32),  # Increased capacity
            nn.ReLU()
        )
        self.lung_head = nn.Linear(32, 2)
        self.heart_head = nn.Linear(32, 2)
        self.kidney_head = nn.Linear(32, 2)

    def forward(self, x, task):
        x = self.shared(x)
        if task == 'lung':
            return self.lung_head(x)
        elif task == 'heart':
            return self.heart_head(x)
        elif task == 'kidney':
            return self.kidney_head(x)
        else:
            raise ValueError("Invalid task")

# === Training Functions ===
def train(model, loader, task, epochs=20):
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.01)
    if task == 'heart':
      weights = torch.tensor([0.7, 1.3]).to(device)  # adjust based on imbalance
    elif task == 'kidney':
      weights = torch.tensor([0.5, 0.5]).to(device)
    else:  # lung
      weights = torch.tensor([0.05,0.95]).to(device)  # since YES is 3% only

    loss_fn = nn.CrossEntropyLoss(weight=weights)


    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x, task), y)
            loss.backward()
            opt.step()

def evaluate(model, loader, task):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x, task).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def compute_fisher(model, loader, task):
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        loss = F.cross_entropy(model(x, task), y)
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data.pow(2)
    for n in fisher:
        fisher[n] /= len(loader)
    return fisher

def ewc_train(model, old_params, fisher, loader, task, lambda_=10, epochs=20):
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.01)
    if task == 'heart':
      weights = torch.tensor([0.7, 1.3]).to(device)  # adjust based on imbalance
    elif task == 'kidney':
      weights = torch.tensor([0.5, 0.5]).to(device)
    else:  # lung
       weights = torch.tensor([0.05,0.95]).to(device)  # since YES is 3% only

    loss_fn = nn.CrossEntropyLoss(weight=weights)


    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x, task), y)
            ewc_penalty = 0
            for n, p in model.named_parameters():
                ewc_penalty += (fisher[n] * (p - old_params[n]).pow(2)).sum()
            loss += (lambda_ / 2) * ewc_penalty
            loss.backward()
            opt.step()

# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Data ===
lung_train_loader = DataLoader(CSVDataset("lung_train.csv", fit=True), batch_size=16, shuffle=True)
heart_train_loader = DataLoader(CSVDataset("heart_train.csv"), batch_size=16, shuffle=True)
kidney_train_loader = DataLoader(CSVDataset("kidney_train.csv"), batch_size=16, shuffle=True)

lung_test_loader = DataLoader(CSVDataset("lung_train.csv"), batch_size=32)
heart_test_loader = DataLoader(CSVDataset("heart_train.csv"), batch_size=32)
kidney_test_loader = DataLoader(CSVDataset("kidney_train.csv"), batch_size=32)

# === Initialize Model ===
model = MultiHeadHospitalNet(input_size=7).to(device)

# === Train & EWC ===
train(model, lung_train_loader, 'lung')
acc_lung_before = evaluate(model, lung_test_loader, 'lung')

lung_params = {n: p.clone().detach() for n, p in model.named_parameters()}
lung_fisher = compute_fisher(model, lung_train_loader, 'lung')

ewc_train(model, lung_params, lung_fisher, heart_train_loader, 'heart')
acc_lung_after_heart = evaluate(model, lung_test_loader, 'lung')
acc_heart = evaluate(model, heart_test_loader, 'heart')

heart_params = {n: p.clone().detach() for n, p in model.named_parameters()}
heart_fisher = compute_fisher(model, heart_train_loader, 'heart')

combined_fisher = {n: lung_fisher[n] + heart_fisher[n] for n in lung_fisher}
combined_params = heart_params

ewc_train(model, combined_params, combined_fisher, kidney_train_loader, 'kidney')

acc_lung_final = evaluate(model, lung_test_loader, 'lung')
acc_heart_final = evaluate(model, heart_test_loader, 'heart')
acc_kidney = evaluate(model, kidney_test_loader, 'kidney')


print("\n🧠 Final Accuracy Report:")
print(f"Lung Accuracy After Lung Training         : {acc_lung_before:.2f}")
print(f"Lung Accuracy After Heart Training (EWC)  : {acc_lung_after_heart:.2f}")
print(f"Heart Accuracy After Heart Training       : {acc_heart:.2f}")
print(f"Lung Accuracy After Kidney Training (EWC) : {acc_lung_final:.2f}")
print(f"Heart Accuracy After Kidney Training (EWC): {acc_heart_final:.2f}")
print(f"Kidney Accuracy After Kidney Training     : {acc_kidney:.2f}")

torch.save(model.state_dict(), "trained_model.pth")
