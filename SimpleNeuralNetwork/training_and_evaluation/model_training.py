import torch

from models.model import NeuralNetwork
from scripts.download_mnist_dataset import create_dataloader
from scripts.save_and_load_model import save_model
from scripts.training_devices import get_training_device


# Create a single training loop
def run_single_training_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            

# Create a single test loop to ensure model is learning
def run_single_test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, " + \
        f"Avg loss: {test_loss:>8f} \n")
    

# Create a train function
def train_model(epochs: int, batch_size: int, save=False, model_name=""):
    # Create a train and test dataloader
    train_dataloader, test_dataloader = create_dataloader(batch_size)

    # Get cpu, gpu or mps device for training.
    device = get_training_device()
    print(f"Using {device} device for training the model.")

    # Create a model
    model = NeuralNetwork().to(device)
    print(f"\nModel: {model}")

    loss_fn = model.create_cross_entropy_loss()  # Create a cross entropy loss function
    optimizer = model.create_sgd_optimizer()  # Create a SGD optimizer

    print("\nStarting Training")
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        run_single_training_loop(train_dataloader, model, loss_fn, optimizer, device)
        run_single_test_loop(test_dataloader, model, loss_fn, device)
        
    print("Finished Training\n")

    if save:
        if model_name == "":
            save_model(model, "model.pth")
        else:
            save_model(model, model_name)    
    print("Done!\n")

    
if __name__ == "__main__":
    train_model(
        epochs=20, batch_size=64, 
        save=True, model_name="my_first_model.pth")    