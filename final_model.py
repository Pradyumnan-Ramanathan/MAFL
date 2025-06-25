# final_model.py

from model_training import Node, load_and_prepare
import pickle

def secure_gossip_training_dh(num_nodes=3, num_rounds=3):
    # Load datasets and scalers
    heart_X, heart_y, heart_scaler = load_and_prepare('heart_dataset_1000.csv')
    kidney_X, kidney_y, kidney_scaler = load_and_prepare('kidney_dataset_1000.csv')
    lung_X, lung_y, lung_scaler = load_and_prepare('lung_dataset_1000.csv')

    scalers = [heart_scaler, kidney_scaler, lung_scaler]
    datasets = [(heart_X, heart_y), (kidney_X, kidney_y), (lung_X, lung_y)]
    nodes = [Node(i, datasets[i][0], datasets[i][1]) for i in range(num_nodes)]

    nodes[0].train_model()
    nodes[0].estimate_importance()

    current_holder = 0
    total_transfers = num_rounds * num_nodes

    for t in range(total_transfers):
        sender = nodes[current_holder]
        receiver = nodes[(current_holder + 1) % num_nodes]

        shared_key_send = sender.compute_shared_key(receiver.public_key)
        shared_key_recv = receiver.compute_shared_key(sender.public_key)
        assert shared_key_send == shared_key_recv

        enc_weights, enc_intercept = sender.encrypt_model(sender.model_weights, sender.intercept, shared_key_send)
        receiver.decrypt_model(enc_weights, enc_intercept, shared_key_recv)

        receiver.train_model()

        if receiver.ewc_importance is None:
            receiver.estimate_importance()

        current_holder = (current_holder + 1) % num_nodes

    return nodes, scalers

if __name__ == "__main__":
    nodes, scalers = secure_gossip_training_dh()
    with open("trained_nodes.pkl", "wb") as f:
        pickle.dump((nodes, scalers), f)

    # ✅ Debug check
    print(len(nodes), len(scalers))
    print(scalers[0].mean_, scalers[0].scale_)
