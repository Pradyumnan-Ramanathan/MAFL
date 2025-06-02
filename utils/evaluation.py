import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

def evaluate_global_model(clients):
    for client in clients:
        clf = SGDClassifier()
        clf.coef_ = client.model_weights.reshape(1, -1)
        clf.intercept_ = np.array([0])
        clf.classes_ = np.array([0, 1])

        preds = clf.predict(client.data)
        acc = accuracy_score(client.labels, preds)
        print(f"Client {client.id} Accuracy: {acc:.2f}")
