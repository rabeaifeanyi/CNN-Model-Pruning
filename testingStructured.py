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

def prune_lns_structured(model, amount, dim):
    # Prune the model using LNS (Lp Norm Soft) structured pruning
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=dim)

    return model

def prune_random_structured(model, amount, dim):
    # Prune the model using random structured pruning
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.random_structured(module, name='weight', amount=amount, dim=dim)

    return model

def train_model(model, train_loader, val_loader):
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    model.to(device)
    epochs = 4

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        start_time = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        end_time = time.time()
        elapsed_time = end_time - start_time
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


        
        print("Epoch: {}/{} | Train Loss: {:.4f} | Validation Loss: {:.4f} | Accuracy: {:.2f}% | Elapased Time: {:.2f}s".format(
            epoch+1, epochs, train_loss, val_loss, accuracy * 100, f1_score_value, elapsed_time))
    
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

    print(accuracy)
    #print(confusion_mat)


if __name__ == '__main__':
    # Load trained unpruned model
    model = CNN()
    model.load_state_dict(torch.load('Models/model.pth'))

    # Create a test DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST(root='Dataset/', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluate the original unpruned model
    print("Original Unpruned Model:")
    eval_model(model, test_loader)

    # Create two instances of the unpruned model and prune them using structured pruning
    model_random_structured = CNN()
    model_random_structured.load_state_dict(torch.load('Models/model.pth'))
    model_random_structured = prune_random_structured(model_random_structured, amount=0.3, dim=0)

    model_lns_structured = CNN()
    model_lns_structured.load_state_dict(torch.load('Models/model.pth'))
    model_lns_structured = prune_lns_structured(model_lns_structured, amount=0.3, dim=0)

    # Evaluate pruned models
    print("\nPruned Models:")
    print("Random Pruning:")
    eval_model(model_random_structured, test_loader)
    print("Magnitude Pruning:")
    eval_model(model_lns_structured, test_loader)

    # Train the pruned models with cross-validation and output mean f1 score, standard deviation of f1 score, and mean accuracy
    # Perform cross-validation split and prepare the data for training
    dataset = datasets.MNIST(root='Dataset/', train=True, transform=transform)
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
        print("Training random pruned model")
        metrics_random = train_model(model_random_structured , train_loader, val_loader)
        print("Trainig magnitude pruned model")
        metrics_magnitude = train_model(model_lns_structured, train_loader, val_loader)

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
    torch.save(model_random_structured .state_dict(), os.path.join('Models', 'model_random.pth'))
    torch.save(model_lns_structured.state_dict(), os.path.join('Models', 'model_magnitude.pth'))

    # Store the results in a text file
    with open('results.txt', 'w') as file:
        file.write("Cross-Validation Results:\n")
        file.write("Random Pruning:\n")
        file.write("Mean F1 Score: {:.4f}\n".format(mean_f1_score_random))
        file.write("Standard Deviation of F1 Score: {:.4f}\n".format(std_f1_score_random))
        file.write("Mean Random Pruned Accuracy: {:.4f}\n".format(mean_accuracy_random))
        file.write("Mean Random Pruned Training Time: {:.4f}\n".format(mean_time_random_training))
        file.write("\n")
        file.write("Magnitude Pruning:\n")
        file.write("Mean F1 Score: {:.4f}\n".format(mean_f1_score_magnitude))
        file.write("Standard Deviation of F1 Score: {:.4f}\n".format(std_f1_score_magnitude))
        file.write("Mean Accuracy: {:.4f}\n".format(mean_accuracy_magnitude))
        file.write("Mean Magnitude Pruned Training Time: {:.4f}\n".format(mean_time_magnitude_training))
    print("Results saved to 'results.txt' file.")

    print("\nCross-Validation Results:")

    print("Mean F1 Score Random Pruning: {:.4f}".format(mean_f1_score_random))
    print("Standard Deviation of F1 Score Random Pruning: {:.4f}".format(std_f1_score_random))
    print("Mean Accuracy Random Pruning: {:.4f}".format(mean_accuracy_random))
    print("Mean Random Pruned Training Time: {:.4f}".format(mean_time_random_training))

    print("Mean F1 Score Magnitude Pruning: {:.4f}".format(mean_f1_score_magnitude))
    print("Standard Deviation of F1 Score Magnitude Pruning: {:.4f}".format(std_f1_score_magnitude))
    print("Mean Accuracy Magnitude Pruning: {:.4f}".format(mean_accuracy_magnitude))
    print("Mean Magnitude Pruned Training Time: {:.4f}".format(mean_time_magnitude_training))
