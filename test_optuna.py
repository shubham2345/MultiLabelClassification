import argparse
import test_optuna
from processing import transform
from sklearn.model_selection import train_test_split
from model import MLPModel
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

def objective(trial):
    # Hyperparameters to optimize
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-1)

    # Load data
    input_vectorized_df, label_encoded_df, vectorizer = transform(training_data_path)
    x_train, x_val, y_train, y_val = train_test_split(input_vectorized_df, label_encoded_df, test_size=0.1, random_state=42)

    # Convert DataFrames to PyTorch tensors
    x_train = torch.FloatTensor(x_train.values)
    y_train = torch.FloatTensor(y_train.values)
    x_val = torch.FloatTensor(x_val.values)
    y_val = torch.FloatTensor(y_val.values)

    # Create TensorDataset and DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize model
    model = MLPModel(input_dim=x_train.shape[1], output_dim=y_train.shape[1], dropout_rate=dropout_rate)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training Loop
    num_epochs = 60
    for epoch in range(num_epochs):
        print("epoch", epoch)
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(x_val)
        val_predictions = (val_outputs > 0.2).int()
        val_accuracy = accuracy_score(y_val.numpy(), val_predictions.numpy())

    return val_accuracy

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Process training and test data.')
    parser.add_argument('training_data', type=str, help='Path to the training data file (CSV, JSON, etc.)')
    parser.add_argument('test_data', type=str, help='Path to the test data file (CSV, JSON, etc.)')
    parser.add_argument('output', type=str, help='Path to the output file where results will be saved')

    # Parse the arguments
    args = parser.parse_args()
    global training_data_path
    training_data_path = args.training_data

    # Create a study object and optimize
    study = test_optuna.create_study(direction="maximize")
    print("hi")
    study.optimize(objective, n_trials=100)

    # Print the best parameters and value
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == "__main__":
    main()
