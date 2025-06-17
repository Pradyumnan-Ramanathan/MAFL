import numpy as np
from sklearn.datasets import make_classification

from encryption.keygen import generate_keys
from clients.client import Client
from server.server import Server
from utils.evaluation import evaluate_global_model

def federated_training(num_clients=3, num_rounds=3):
    X, y = make_classification(n_samples=300, n_features=5, n_informative=3, n_classes=2)
    split_data = np.array_split(X, num_clients)
    split_labels = np.array_split(y, num_clients)

    public_key, private_key = generate_keys()

    clients = [Client(i, public_key, private_key, split_data[i], split_labels[i]) for i in range(num_clients)]
    server = Server(public_key)

    for round in range(num_rounds):
        print(f"\n🔁 Round {round + 1} - Local training and encryption")
        server.encrypted_models.clear()

        for client in clients:
            client.train_local_model()
            enc_weights = client.encrypt_weights()
            server.receive_encrypted_model(enc_weights)
            print(f"Client {client.id} encrypted and sent weights.")

        print("🔐 Server aggregating encrypted models...")
        encrypted_global_model = server.aggregate_encrypted_models(num_clients)

        for client in clients:
            decrypted_global_model = client.decrypt_weights(encrypted_global_model)
            client.model_weights = decrypted_global_model
            print(f"Client {client.id} decrypted global model.")

    return clients

if __name__ == "__main__":
    clients = federated_training()
    evaluate_global_model(clients)
