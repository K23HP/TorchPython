from training_and_evaluation.model_evaluation import evaluate_model
from training_and_evaluation.model_training import train_model

model_name = "mnist_classification_model.pth"


def main():
    # Start training the model
    train_model(
        epochs=1,
        batch_size=64,
        save=True,
        model_name=model_name
    )

    # Evaluate on the trained model
    evaluate_model(model_name)


if __name__ == "__main__":
    main()
