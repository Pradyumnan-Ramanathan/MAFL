
import numpy as np
from sklearn.linear_model import SGDClassifier

class Client:
    def __init__(self, id, public_key, private_key, data, labels):
        self.id = id
        self.public_key = public_key
        self.private_key = private_key
        self.data = data
        self.labels = labels
        self.model_weights = None
    
    def train_local_model(self):
        clf = SGDClassifier(loss='log_loss', max_iter=1000, learning_rate='constant', eta0=0.01)
        clf.fit(self.data, self.labels)
        self.model_weights = clf.coef_[0]
        return self.model_weights

    def encrypt_weights(self):
        return [self.public_key.encrypt(x) for x in self.model_weights]
    
    def decrypt_weights(self, encrypted_weights):
        return np.array([self.private_key.decrypt(x) for x in encrypted_weights])
