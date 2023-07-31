import torch


# Create a function to get cpu, gpu or mps device for training
def get_training_device():
    if torch.cuda.is_available():
        return "cuda:0"
    else:
        return "mps:0" if torch.backends.mps.is_available() else "cpu"
