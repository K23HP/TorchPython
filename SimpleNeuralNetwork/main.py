from training_and_evaluation.model_evaluation import evaluate_model
from training_and_evaluation.model_training import train_model


def main():
    # Start training the model
    train_model(
        epochs=20, 
        batch_size=64, 
        save=True, 
        model_name="my_new_model.pth"
    )
    
    # Evaluate on the trained model
    evaluate_model("model.pth")

if __name__ == "__main__":
    main()