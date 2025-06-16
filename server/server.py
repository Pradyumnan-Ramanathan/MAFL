class Server:
    def __init__(self, public_key):
        self.public_key = public_key
        self.encrypted_models = []

    def receive_encrypted_model(self, encrypted_model):
        self.encrypted_models.append(encrypted_model)

    def aggregate_encrypted_models(self, num_clients):
        n_weights = len(self.encrypted_models[0])
        aggregated_model = []
        for i in range(n_weights):
            sum_enc = sum(client_weights[i] for client_weights in self.encrypted_models)
            avg_enc = sum_enc * (1 / num_clients)
            aggregated_model.append(avg_enc)
        return aggregated_model
