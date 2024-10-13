import argparse
from processing import transform, training_validation_dataset
from sklearn.model_selection import train_test_split
from model import MLPModel
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import StepLR



def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Process training and test data.')

    # Add positional arguments for training_data, test_data, and output
    parser.add_argument('training_data', type=str,
                        help='Path to the training data file (CSV, JSON, etc.)')
    parser.add_argument('test_data', type=str,
                        help='Path to the test data file (CSV, JSON, etc.)')
    parser.add_argument('output', type=str,
                        help='Path to the output file where results will be saved')

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    training_data_path = args.training_data
    test_data_path = args.test_data
    output_path = args.output

    # Print the arguments for verification (optional)
    print(f'Training Data Path: {training_data_path}')
    print(f'Test Data Path: {test_data_path}')
    print(f'Output Path: {output_path}')
 
    # Transform and split the training data
    input_vectorized_df, label_encoded_df, vectorizer, count_vectorizer_scalar, word2vec_scalar = transform(training_data_path)


    # Split the dataset into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(input_vectorized_df, label_encoded_df, test_size=0.005, random_state=42)

    print(x_train.shape, x_val.shape)
    print(y_train.shape, y_val.shape)






    # Initialize model, loss function, and optimizer
    # Split the dataset into training and validation sets
    # print("bois", x_train, x_train.values)
    # Convert DataFrames to PyTorch tensors
    x_train = torch.FloatTensor(x_train.values)
    # print(type(x_train))
    y_train = torch.FloatTensor(y_train.values) # Ensure correct shape
    x_val = torch.FloatTensor(x_val.values)
    y_val = torch.FloatTensor(y_val.values)  # Ensure correct shape

    # Normalize your input data for training
    # mean = x_train.mean(dim=0)
    # std = x_train.std(dim=0)

    # x_train = (x_train - mean) / std
    # x_val = (x_val - mean) / std


    # Create TensorDataset and DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Set batch_size here


    model = MLPModel(input_dim=x_train.shape[1], output_dim=y_train.shape[1])  # Adjust input_dim and output_dim
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    # print(model, criterion, optimizer)

    # Initialize the learning rate scheduler
    # scheduler = StepLR(optimizer, step_size=50, gamma=0.1)  # Reduce learning rate every 50 epochs by a factor of 0.1


    epochs=[]
    training_accuracy=[]
    training_loss=[]
    validation_accuracy=[]
    validation_loss = []
    # Training Loop
    num_epochs=200
    patience = 10  # Number of epochs to wait for improvement
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        epochs.append(epoch)
        model.train()  # Set the model to training mode
        batch_training_loss = []
        batch_validation_loss = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(batch_x) # Forward pass
            loss = criterion(outputs, batch_y)  # Compute the loss
            batch_training_loss.append(loss.item())
            # print("Training Loss", loss.item())
            # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
        training_loss.append(sum(batch_training_loss)/len(batch_training_loss))

        model.eval()
        with torch.no_grad():
            train_outputs = model(x_train)
            # print(train_outputs[0])
            train_predictions = (train_outputs > 0.5).int()  # Binarize predictions
            # print(train_predictions[0])
            train_accuracy = accuracy_score(y_train.numpy(), train_predictions.numpy())
            training_accuracy.append(train_accuracy)
            #rain_accuracy = (train_predictions == y_train.int()).sum().item() / len(y_train)
    # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Training Accuracy: {train_accuracy:.4f}')



        # model.eval()
        # with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
            batch_validation_loss.append(val_loss)
            # print(val_loss.item())
            val_predictions = (val_outputs > 0.5).int()
            # (predicted)
            val_accuracy = accuracy_score(y_val.numpy(), val_predictions.numpy())
            validation_accuracy.append(val_accuracy)
            # acc = (predicted == y_val.int()).sum() / len(y_val)
            # print("Accuracy on the validation dataset", val_accuracy)

            # Check if we have a new best validation loss
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     epochs_without_improvement = 0  # Reset the counter
            #     # You could also save the model here if you want
            # else:
            #     epochs_without_improvement += 1
            
            # # Check if we should stop training
            # if epochs_without_improvement >= patience:
            #     print(f"Early stopping triggered at epoch {epoch + 1}")  # Show which epoch it stopped
            #     break

        validation_loss.append(sum(batch_validation_loss)/len(batch_validation_loss))

        # scheduler.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    #plotting 
    epochs=np.array(epochs)
    plt.figure(figsize=(12, 5))

    # Plotting training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_loss, label='Training Loss', marker='o', color='blue')
    plt.plot(epochs, validation_loss, label='Validation Loss', marker='o', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.grid()

    # Plotting validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_accuracy, label='Training Accuracy', marker='o', color='blue')
    plt.plot(epochs, validation_accuracy, label='Validation Accuracy', marker='o', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


    # Your mainn logic here
    # For example: load data, train a model, and save output

    test_input_vectorized_df, _ = transform(test_data_path, vectorizer=vectorizer, count_vectorizer_scaler=count_vectorizer_scalar, word2vec_scaler=word2vec_scalar)  # Only transform inputs
    print(test_input_vectorized_df.shape)
    # Convert test data to PyTorch tensor
    x_test_tensor = torch.FloatTensor(test_input_vectorized_df.values)

    # x_test_tensor = (x_test_tensor - mean) / std 


    # Make predictions on the test data
    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test_tensor)
        test_predictions = (test_outputs > 0.5).int()  # Binarize predictions

    
    class_names = ['actor.gender', 'gr.amount', 'movie.country', 'movie.directed_by', 'movie.estimated_budget', 
                   'movie.genre', 'movie.gross_revenue', 'movie.initial_release_date', 'movie.language', 
                   'movie.locations', 'movie.music', 'movie.produced_by', 'movie.production_companies', 
                   'movie.rating', 'movie.starring.actor', 'movie.starring.character', 'movie.subjects', 
                   'none', 'person.date_of_birth']  # Get column names for relations
    results = []

    # Iterate over predictions to construct the output
    for index in range(test_predictions.shape[0]):
        predicted_classes = [class_names[i] for i in range(len(class_names)) 
                             if test_predictions[index, i] == 1]
        results.append({
            'ID': pd.read_csv(test_data_path)['ID'].iloc[index],  # Get corresponding ID
            'CORE RELATIONS': ' '.join(predicted_classes)  # Join class names with spaces
        })

    # Create a DataFrame for the results
    results_df = pd.DataFrame(results)

    # Save to CSV
    results_df.to_csv(output_path, index=False)  # Save predictions to output file
    print("Predictions saved to", output_path)


if __name__ == "__main__":
    main()
