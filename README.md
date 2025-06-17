# 🛡️ Privacy-Preserving Federated Learning (PPFL) with Paillier Encryption

This project implements **two-level privacy** in **federated learning** using **Paillier Homomorphic Encryption (PHE)** to detect CIoT attacks using encrypted model weights.

## 🔧 Features

- Local training on clients using `SGDClassifier`
- Paillier-based encryption and averaging of weights
- Federated Averaging (FedAvg) on encrypted weights
- Performance evaluation post-decryption

## 📁 Structure

- `clients/`: Handles local training and encryption
- `server/`: Aggregates encrypted model updates
- `encryption/`: Key generation and Paillier logic
- `utils/`: Accuracy evaluation

## 🚀 Getting Started

```bash
git clone https://github.com/yourusername/privacy-preserving-fl-ciot.git
cd privacy-preserving-fl-ciot
pip install -r requirements.txt
python main.py
