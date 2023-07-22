from datetime import datetime


def get_current_date_time():
    current_datetime = datetime.now()
    year = current_datetime.year
    month = current_datetime.month
    day = current_datetime.day
    hour = current_datetime.hour
    minute = current_datetime.minute

    return year, month, day, hour, minute


# Save train and test loss in csv
def save_train_losses(train_losses: list, test_losses: list):
    year, month, day, hour, minute = get_current_date_time()

    losses = {
        "Training Loss": train_losses,
        "Test Loss": test_losses,
        }

    # df = pd.DataFrame(losses)
    # file_name = f"train_losses_{year}_{month}_{day}_{hour}_{minute}.csv"
    # log_path = f"./logs/{file_name}"
    # df.to_csv(log_path, index=False)
    print(losses)
