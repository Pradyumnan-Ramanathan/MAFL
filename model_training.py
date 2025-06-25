# model_training.py

import numpy as np
import pandas as pd
import random
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

q = 7919
alpha = 2

def modular_pow(base, exponent, modulus):
    result = 1
    base = base % modulus
    while exponent > 0:
        if exponent % 2:
            result = (result * base) % modulus
        exponent = exponent >> 1
        base = (base * base) % modulus
    return result

class Node:
    def __init__(self, id, data, labels):
        self.id = id
        self.private_key = random.randint(2, q - 2)
        self.public_key = modular_pow(alpha, self.private_key, q)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        self.model_weights = None
        self.intercept = 0
        self.ewc_lambda = 100
        self.ewc_importance = None
        self.optimal_weights = None

    def compute_shared_key(self, other_public_key):
        return modular_pow(other_public_key, self.private_key, q)

    def encrypt_model(self, weights, intercept, shared_key):
        enc_weights = [w + shared_key for w in weights]
        enc_intercept = intercept + shared_key
        return enc_weights, enc_intercept

    def decrypt_model(self, enc_weights, enc_intercept, shared_key):
        self.model_weights = np.array([w - shared_key for w in enc_weights])
        self.intercept = enc_intercept - shared_key

    def estimate_importance(self):
        clf = SGDClassifier(loss='log_loss', max_iter=1, learning_rate='constant', eta0=0.01, random_state=42)
        clf.partial_fit(self.X_train, self.y_train, classes=[0, 1])
        grads = clf.coef_[0]
        self.ewc_importance = grads ** 2
        self.optimal_weights = np.copy(self.model_weights)

    def train_model(self):
        clf = SGDClassifier(loss='log_loss', max_iter=1000, learning_rate='constant', eta0=0.01, random_state=42)
        if self.model_weights is not None:
            clf.coef_ = self.model_weights.reshape(1, -1)
            clf.intercept_ = np.array([self.intercept])
            clf.classes_ = np.array([0, 1])
        clf.partial_fit(self.X_train, self.y_train, classes=[0, 1])
        updated_weights = clf.coef_[0]
        updated_intercept = clf.intercept_[0]
        if self.ewc_importance is not None and self.optimal_weights is not None:
            delta = updated_weights - self.optimal_weights
            correction = 0.01 * self.ewc_lambda * self.ewc_importance * delta
            updated_weights -= correction
        self.model_weights = updated_weights
        self.intercept = updated_intercept

    def get_predictor(self):
        clf = SGDClassifier(loss='log_loss')  # ← Add log_loss here
        clf.coef_ = self.model_weights.reshape(1, -1)
        clf.intercept_ = np.array([self.intercept])
        clf.classes_ = np.array([0, 1])
        return clf


    def evaluate(self):
        clf = self.get_predictor()
        preds = clf.predict(self.X_test)
        return accuracy_score(self.y_test, preds)

def load_and_prepare(filepath):
    df = pd.read_csv(filepath)
    df.columns.values[-1] = 'target'
    y = df['target'].values
    X = df.drop(columns=['target']).values
    ros = RandomOverSampler(random_state=42)
    X, y = ros.fit_resample(X, y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, scaler
