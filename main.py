from model_evaluation import evaluate_model
from training.model_training import train_model


def main():
    # Start training the model
    train_model(epochs=20, batch_size=64, save=False)
    
    # Evaluate on the trained model
    evaluate_model()

if __name__ == "__main__":
    main()