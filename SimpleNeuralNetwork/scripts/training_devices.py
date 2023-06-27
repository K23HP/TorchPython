import torch


# Create a function to get cpu, gpu or mps device for training
def get_training_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )