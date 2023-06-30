import torch
from models.model import NeuralNetwork

from scripts.training_devices import get_training_device


# Save the trained model
def save_model(model, model_name):
    torch.save(model.state_dict(), f"./pre-trained_models/{model_name}")
    print(f"Saved PyTorch Model State to '{model_name}'.")


# Load the trained model
def load_model(model_name):
    device = get_training_device()
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(f"./pre-trained_models/{model_name}"))
    return model.eval()
