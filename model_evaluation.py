import torch

from scripts.download_mnist_dataset import get_mnist_test_data
from scripts.save_and_load_model import load_model
from scripts.training_devices import get_training_device

# Create a list of classes for classification
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

def evaluate_model(model_name="my_first_model.pth"):
    """Evaluate the model performance

    Args:
        model_name (str, optional): Name of the model to evaluate. 
        Defaults to "my_first_model.pth".
    """
    # device = get_training_device()
    # model = NeuralNetwork().to(device)
    # model.load_state_dict(torch.load(f"./trained_models/{model_name}"))
    
    model = load_model(model_name)  # Load model to evaluate
    test_data = get_mnist_test_data()  # Load test_data

    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        print("<============ Model evaluation result ============>")
        make_prediction(x, y, model)


def make_prediction(x, y, model):
    """Report the model predicted item and actual item in mnist testset."""
    device = get_training_device()
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'>>> Predicted: "{predicted}", Actual: "{actual}"')


if __name__ == "__main__":
    evaluate_model()

