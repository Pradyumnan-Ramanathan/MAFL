import streamlit as st
import torch
import torch.nn as nn
import joblib

# === Load Scaler ===
scaler = joblib.load("scaler.pkl")

# === Define Model ===
class MultiHeadHospitalNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, 32),  # Must match training
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

# === Load Trained Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiHeadHospitalNet(input_size=7).to(device)
model.load_state_dict(torch.load("trained_model.pth", map_location=device))
model.eval()

# === Streamlit UI ===
st.title("Disease Prediction App")
selected_task = st.selectbox("Select Disease Type", ['lung', 'heart', 'kidney'])

st.subheader(f"Enter 7 Features for {selected_task.capitalize()} Patient")
feature_names = ['age', 'bp', 'cholesterol', 'feature4', 'feature5', 'feature6', 'feature7']
inputs = []
for name in feature_names:
    val = st.number_input(f"{name}", value=0.0, step=0.1)
    inputs.append(val)


if st.button("Predict"):
    scaled_input = scaler.transform([inputs])
    x_input = torch.tensor([inputs], dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(x_input, selected_task)
        prob = torch.softmax(output, dim=1)[0][1].item()  # probability of class 1 (YES)

# Adjust the threshold if needed (e.g., 0.4 to be more sensitive)
        prediction = 1 if prob > 0.4 else 0


    if prediction == 0:
        st.success(f"{selected_task.capitalize()} Disease: NO (Negative)")
    else:
        st.error(f"{selected_task.capitalize()} Disease: YES (Positive)")
