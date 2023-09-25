import torch
import os
import time
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import streamlit as st
import csv


# Define your CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # Forward pass through the conv layers
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the feature maps
        # Forward pass through the fully connected layers
        x = self.fc_layers(x)
        return x

def prune_magnitude(model, pruning_perc):
    # Prune the model using magnitude-based pruning
    parameters_to_prune = [(module, 'weight') for module in model.modules() if isinstance(module, nn.Conv2d)]
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_perc
    )
    return model

def prune_random(model, pruning_perc):
    # Apply random pruning to the model
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.random_unstructured(module, name='weight', amount=pruning_perc)

    # Prune the weight parameter of the model
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.remove(module, 'weight')

    return model

def train_model(model, train_loader, val_loader):
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    model.to(device)
    epochs = 4
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        predictions = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = correct / total
        f1_score_value = f1_score(torch.cat([labels for _, labels in val_loader]), predictions, average='macro')


        
        st.text("Epoch: {}/{} | Train Loss: {:.4f} | Validation Loss: {:.4f} | Accuracy: {:.2f}%".format(
            epoch+1, epochs, train_loss, val_loss, accuracy * 100, f1_score_value))
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    st.text("Elapsed Time: {:.2f}s".format(elapsed_time))
    # Evaluate the model on the validation set and return the metrics
    metrics = {'loss': val_loss, 'accuracy': accuracy, 'f1_score': f1_score_value, 'elapsed_time': elapsed_time}
    return metrics

def eval_model(model, dataloader):
    # Define transformations to apply to the data
    transform = transforms.Compose([
    transforms.ToTensor(),               # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize the pixel values to the range [-1, 1]
    ])


    # Set the model to evaluation mode
    model.eval()
    # Initialize lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        correct = 0
        total = 0
        # Iterate over the test dataset and make predictions
        for images, labels in dataloader:
            # Forward pass through the model
            outputs = model(images)

            # Get the predicted labels by finding the maximum value along the class dimension
            _, predicted = torch.max(outputs, 1)

            # Store the true and predicted labels
            true_labels.extend(labels.numpy())
            predicted_labels.extend(predicted.numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        

    # Calculate accuracy and the confusion matrix
    accuracy = correct / total
    confusion_mat = confusion_matrix(true_labels, predicted_labels)

    return accuracy
    #st.text(confusion_mat)

    
def run(model_path, dataset, dataset_path, session):
    # Load trained unpruned model
    model = CNN()
    model.load_state_dict(torch.load(model_path))
    
    # Create a test DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if dataset ==  "MNIST":
        test_dataset = datasets.MNIST(root=dataset_path, train=False, transform=transform, download=True)
        
    else:
        #TODO different dataset
        test_dataset = datasets.MNIST(root=dataset_path, train=False, transform=transform, download=True)
        
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluate the original unpruned model
    accuracy_original = eval_model(model, test_loader)
    st.text(f"Accuracy of Original Model: {accuracy_original}")

    # Create two instances of the unpruned model and prune them
    model_random = CNN()
    model_random.load_state_dict(torch.load(model_path))
    model_random = prune_random(model_random, 0.3)

    model_magnitude = CNN()
    model_magnitude.load_state_dict(torch.load(model_path))
    model_magnitude = prune_magnitude(model_magnitude, 0.3)

    # Evaluate pruned models
    st.text("\nPruned Models:")
    accuracy_random = eval_model(model_random, test_loader)
    st.text(f"Accuracy of Randomly Pruned Model: {accuracy_random}")
    accuracy_magnitude = eval_model(model_magnitude, test_loader)
    st.text(f"Accuracy of Original Model: {accuracy_magnitude}")

    # Train the pruned models with cross-validation and output mean f1 score, standard deviation of f1 score, and mean accuracy
    # Perform cross-validation split and prepare the data for training
    dataset = datasets.MNIST(root=dataset_path, train=True, transform=transform)
    num_folds = 3
    num_samples = len(dataset)
    fold_size = num_samples // num_folds

    # Randomly shuffle the dataset
    indices = np.random.permutation(num_samples)

    f1_scores_random = []
    f1_scores_magnitude = []
    accuracies_random = []
    accuracies_magnitude = []
    time_random_training = []
    time_magnitude_training = []

    for fold in range(num_folds):
        # Split the dataset into training and validation sets
        val_indices = indices[fold * fold_size : (fold + 1) * fold_size]
        train_indices = np.concatenate((indices[:fold * fold_size], indices[(fold + 1) * fold_size:]))

        # Create data loaders for training and validation sets
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Train the pruned models
        st.markdown(":red[Please wait, while training random pruned model...]")
        metrics_random = train_model(model_random, train_loader, val_loader)
        
        st.markdown(":red[Please wait, while training magnitude pruned model...]")
        metrics_magnitude = train_model(model_magnitude, train_loader, val_loader)

        # Record the f1 scores, accuracies and time
        f1_scores_random.append(metrics_random['f1_score'])
        f1_scores_magnitude.append(metrics_magnitude['f1_score'])
        accuracies_random.append(metrics_random['accuracy'])
        accuracies_magnitude.append(metrics_magnitude['accuracy'])
        time_random_training.append(metrics_random['elapsed_time'])
        time_magnitude_training.append(metrics_magnitude['elapsed_time'])

    # Calculate the mean f1 score, standard deviation of f1 score, mean accuracy and time
    mean_f1_score_random = np.mean(f1_scores_random)
    mean_f1_score_magnitude = np.mean(f1_scores_magnitude)
    std_f1_score_random = np.std(f1_scores_random)
    std_f1_score_magnitude = np.std(f1_scores_magnitude)
    mean_accuracy_random = np.mean(accuracies_random)
    mean_accuracy_magnitude = np.mean(accuracies_magnitude)
    mean_time_random_training = np.mean(time_random_training)
    mean_time_magnitude_training = np.mean(time_magnitude_training)
    # Save the trained pruned models
    torch.save(model_random.state_dict(), os.path.join('Models', 'model_random.pth'))
    torch.save(model_magnitude.state_dict(), os.path.join('Models', 'model_magnitude.pth'))

    # Store the results in csv file
    with open(f'pyTorchResults{session}.csv', 'w', newline='') as csvfile:
        fieldnames = ['Metric', 'Value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Metric': 'Accuracy Original Model', 'Value': accuracy_original})
        writer.writerow({'Metric': 'Accuracy Randomly Pruned Model', 'Value': accuracy_random})
        writer.writerow({'Metric': 'Accuracy Magnitude Pruned Model', 'Value': accuracy_magnitude})
        writer.writerow({'Metric': 'Mean F1 Score Random Pruning', 'Value': mean_f1_score_random})
        writer.writerow({'Metric': 'Mean F1 Score Magnitude Pruning', 'Value': mean_f1_score_magnitude})
        writer.writerow({'Metric': 'Standard Deviation of F1 Score Random Pruning', 'Value': std_f1_score_random})
        writer.writerow({'Metric': 'Standard Deviation of F1 Score Magnitude Pruning', 'Value': std_f1_score_magnitude})
        writer.writerow({'Metric': 'Mean Random Pruned Training Time', 'Value': mean_time_random_training})
        writer.writerow({'Metric': 'Mean Magnitude Pruned Training Time', 'Value': mean_time_magnitude_training})