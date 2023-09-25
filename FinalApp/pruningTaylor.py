import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import torch.nn.utils.prune as prune
import time
import numpy as np


class TaylorExpansionPruning(prune.BasePruningMethod):
    # Structured pruning implementation

    def __init__(self, amount):
        self.amount = amount  # Fraction of channels to be pruned
    
    def compute_mask(self, t, default_mask):
        """
        Computes a mask for pruning based on Taylor Expansion.

        Parameters:
        - t (Tensor): Output tensor from the activation function.
        - default_mask (Tensor): Default mask tensor.

        Returns:
        - mask (Tensor): The computed pruning mask.
        """

        # Compute the gradients of sum of output tensor t w.r.t input tensor t
        grads = torch.autograd.grad(t.sum(), t, create_graph=True)[0]
        
        # Taylor Expansion: Approximate function near activation as t * grads
        taylor_expansion = t * grads  # <-- Taylor Expansion is performed here
        
        # Compute norms along spatial dimensions (H, W) and along channels (C)
        norms = torch.sum(taylor_expansion**2, dim=(1, 2, 3))
        
        num_channels = t.shape[1]
        num_prune = int(self.amount * num_channels)  # Number of channels to prune

        # Get indices of 'num_prune' smallest norms
        _, prune_indices = torch.topk(norms, num_prune, largest=False)

        # Clone the default mask and update it
        mask = default_mask.clone()
        for idx in prune_indices:
            mask[idx] = 0  # Set corresponding positions in mask to zero

        return mask


def taylor_expansion_pruning(model, name, amount):
    """
    Applies Taylor Expansion-based pruning on Conv2D layers of the model.

    Parameters:
    - model (nn.Module): Neural network model.
    - name (str): Name of the parameter to be pruned: 'weight'.
    - amount (float): Fraction of channels to be pruned.

    """

    for layer in model.children():
        # Check if the layer is Conv2D, then apply Taylor Expansion Pruning
        if isinstance(layer, nn.Conv2d):
            TaylorExpansionPruning.apply(layer, name, amount)


def train_model(model, train_loader, val_loader):
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    model.to(device)
    epochs = 1
    
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


        
        print("Epoch: {}/{} | Train Loss: {:.4f} | Validation Loss: {:.4f} | Accuracy: {:.2f}%".format(
            epoch+1, epochs, train_loss, val_loss, accuracy * 100, f1_score_value))
        
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print("Elapsed Time: {:.2f}s".format(elapsed_time))
    # Evaluate the model on the validation set and return the metrics
    metrics = {'loss': val_loss, 'accuracy': accuracy, 'f1_score': f1_score_value, 'elapsed_time': elapsed_time}

    return metrics

def eval_model(model, dataloader):
    # Define transformations to apply to the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

            images, labels = images.to(device), labels.to(device)

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
    f1_score_value = f1_score(torch.cat([labels for _, labels in dataloader]), predicted_labels, average='macro')

    # Print the accuracy and F1 score
    print("Accuracy: {:.2f}".format(accuracy))
    print("F1 Score: {:.2f}".format(f1_score_value))

    # Return the metrics dictionary
    metrics = {'accuracy': accuracy, 'f1_score': f1_score_value}
    return metrics


def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pretrained ResNet18 model
    resnet_model = models.resnet18(pretrained=False).to(device)

    # Modify the first layer and the last layer for MNIST
    resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False).to(device)
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 10).to(device)

    # Data loading part remains the same
    # Create a test DataLoader
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Mean and Std dev for MNIST
    ])
    
    test_dataset = datasets.MNIST(root='Dataset/', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Prepare the data for retraining
    dataset = datasets.MNIST(root='Dataset/', train=True, transform=transform)
    num_samples = len(dataset)
    num_train = int(0.8 * num_samples)
    train_indices = np.random.permutation(num_samples)[:num_train]
    val_indices = np.setdiff1d(np.arange(num_samples), train_indices)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Train the original, unpruned model
    print("Training Original Unpruned Model")
    train_metrics_original = train_model(resnet_model, train_loader, val_loader)

    # Evaluate the original unpruned model
    print("Original Unpruned Model:")
    val_metrics_original = eval_model(resnet_model, test_loader)
    print(f"Validation Metrics Pruned: {val_metrics_original}")
    # Apply Taylor Expansion-based pruning
    taylor_expansion_pruning(resnet_model, name='weight', amount=0.4)
    print("Taylor Expansion Pruning applied successfully")

    # Retrain and evaluate the pruned model
    print("Evaluating the Taylor Expansion Pruned Model without retrain.")
    val_metrics_pruned = eval_model(resnet_model, test_loader)
    print(f"Validation Metrics Pruned: {val_metrics_pruned}")

    # Train pruned model
    print("Training Custom Structured Taylor Expansion Based Pruned Model")
    retrain_metrics_pruned = train_model(resnet_model, train_loader, val_loader)

    # Evaluate the pruned model after retraining
    print("Evaluation of Custom Structured Taylor Expansion Based Pruned Model")
    val_metrics_pruned_retrain = eval_model(resnet_model, test_loader)
