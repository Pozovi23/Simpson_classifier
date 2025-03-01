import csv


def train_val_get():
    train_path = "simpson_train_set.csv"
    validation_path = "simpson_train_set.csv"
    train = []
    with open(train_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            train.append(line)

    validation = []
    with open(validation_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            validation.append(line)

    return train, validation
