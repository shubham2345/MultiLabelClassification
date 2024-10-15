import argparse
from processing import transform
from sklearn.model_selection import train_test_split
from model import MLPModel
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_curve
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report



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
 
    #count vectorizer with analyzer as char
    input_vectorized_df, label_encoded_df, vectorizer = transform(training_data_path)

    # Split the dataset into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(input_vectorized_df, label_encoded_df, test_size=0.2, random_state=42)

    print(x_train.shape, x_val.shape)
    print(y_train.shape, y_val.shape)



    x_train = torch.FloatTensor(x_train.values)
    y_train = torch.FloatTensor(y_train.values) # Ensure correct shape
    x_val = torch.FloatTensor(x_val.values)
    y_val = torch.FloatTensor(y_val.values)  # Ensure correct shape


    # Create TensorDataset and DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Set batch_size here


    model = MLPModel(input_dim=x_train.shape[1], output_dim=y_train.shape[1])  # Adjust input_dim and output_dim
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=0.0005,weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)


    # Initialize the learning rate scheduler
    # scheduler = StepLR(optimizer, step_size=50, gamma=0.1)  # Reduce learning rate every 50 epochs by a factor of 0.1


    epochs=[]
    training_accuracy=[]
    training_loss=[]
    validation_accuracy=[]
    validation_loss = []
    optimal_thresholds=[]

    # Training Loop
    num_epochs=45
    patience = 10  # Number of epochs to wait for improvement
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print("epoch", epoch)
        epochs.append(epoch)
        model.train()  # Set the model to training mode
        batch_training_loss = []
        batch_validation_loss = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(batch_x) # Forward pass
            loss = criterion(outputs, batch_y)  # Compute the loss
            batch_training_loss.append(loss.item())
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
        training_loss.append(sum(batch_training_loss)/len(batch_training_loss))

        model.eval()
        with torch.no_grad():
            #Training Loss & Accuracy
            train_outputs = model(x_train)
            train_predictions = (train_outputs > 0.38).int()
            train_accuracy = accuracy_score(y_train.numpy(), train_predictions.numpy())
            training_accuracy.append(train_accuracy)
            #Validation Loss & Accuracy
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
            batch_validation_loss.append(val_loss)
            val_predictions = (val_outputs > 0.38).int()
            val_accuracy = accuracy_score(y_val.numpy(), val_predictions.numpy())
            validation_accuracy.append(val_accuracy)

            last_val_outputs=val_outputs.cpu().numpy()
        validation_loss.append(sum(batch_validation_loss)/len(batch_validation_loss))


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


        # scheduler.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    class_names = ['actor.gender', 'gr.amount', 'movie.country', 'movie.directed_by', 'movie.estimated_budget', 
                   'movie.genre', 'movie.gross_revenue', 'movie.initial_release_date', 'movie.language', 
                   'movie.locations', 'movie.music', 'movie.produced_by', 'movie.production_companies', 
                   'movie.rating', 'movie.starring.actor', 'movie.starring.character', 'movie.subjects', 
                   'none', 'person.date_of_birth']
    
        # Use the last_val_outputs for threshold calculations
    optimal_thresholds = []
    for i in range(last_val_outputs.shape[1]):  # Iterate over the number of classes
        precision, recall, thresholds = precision_recall_curve(y_val[:, i].numpy(), last_val_outputs[:, i])

        # Calculate F1 scores and find the optimal threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # Avoid division by zero
        # print(f1_scores.shape)
        
        optimal_idx = np.argmax(f1_scores) if len(f1_scores) > 0 else 0  # Ensure optimal_idx is assigned a value
        optimal_threshold = thresholds[optimal_idx] if len(thresholds) > 0 else 0  # Ensure optimal_threshold is assigned a value
        optimal_thresholds.append(optimal_threshold)

        # plt.figure(figsize=(10, 6))
        # plt.plot(recall, precision, marker='.')
        # plt.axvline(x=recall[optimal_idx], color='red', linestyle='--', label='Optimal Threshold = {:.2f}'.format(optimal_threshold))
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.title(f'Precision-Recall Curve for Class: {class_names[i]}')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.0])
        # plt.grid()
        # plt.legend()
        # plt.show()
        

        print(f'Class {i+1} - Optimal Threshold: {optimal_threshold:.4f}')

    # Calculate the average threshold
    average_threshold = np.mean(optimal_thresholds)
    print(f'Average Threshold: {average_threshold:.4f}')

    # Generate classification report using the last validation outputs
    last_val_predictions = (last_val_outputs > average_threshold).astype(int)  # Binarize based on average threshold
    print("Classification Report:")
    print(classification_report(y_val.numpy(), last_val_predictions, target_names=class_names))

    # You can also save the predictions to a CSV if needed
    val_predictions_df = pd.DataFrame(last_val_outputs, columns=[f'Class_{i + 1}' for i in range(last_val_outputs.shape[1])])
    val_predictions_df['Actual'] = y_val.numpy().tolist()  # Add actual labels for comparison
    val_predictions_df.to_csv('data/val_predictions.csv', index=False)
    print("Validation predictions saved to val_predictions.csv")
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

    test_input_vectorized_df, _ = transform(test_data_path, vectorizer=vectorizer)
    print(test_input_vectorized_df.shape)
    # Convert test data to PyTorch tensor
    x_test_tensor = torch.FloatTensor(test_input_vectorized_df.values)

    # Make predictions on the test data
    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test_tensor)
        test_predictions = (test_outputs > 0.38).int()  # Binarize predictions

    
    
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
