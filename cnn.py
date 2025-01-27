import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

train_set = torchvision.datasets.FashionMNIST(
    root=".", train=True, download=True, transform=transforms.ToTensor()
)
validation_set = torchvision.datasets.FashionMNIST(
    root=".", train=False, download=True, transform=transforms.ToTensor()
)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
validation_loader = torch.utils.data.DataLoader(
    validation_set, batch_size=32, shuffle=False
)

torch.manual_seed(42)


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)

        # Fully connected layers
        self.fc1 = nn.Linear(
            64 * 7 * 7, 1024
        )  # Output of second max pooling is 7x7 with 64 channels
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 10)

        # Initialize weights using Xavier uniform initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Trainer:
    def __init__(self, model, train_loader, test_loader, epochs=30, lr=0.1):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.train_losses = []  # Store the training loss per epoch
        self.val_losses = []  # Store the validation loss per epoch
        self.train_accuracies = []  # Store the training accuracy per epoch
        self.val_accuracies = []  # Store the validation accuracy per epoch

    def train(self):
        """
        Trains the model for the specified number of epochs. It also computes the training and validation loss
        and accuracy at each epoch.

        Parameters:
        - model (nn.Module): The neural network model to be trained.
        - train_loader (DataLoader): The DataLoader object that contains the training data.
        - test_loader (DataLoader): The DataLoader object that contains the validation data.
        - epochs (int): The number of epochs to train the model.
        - lr (float): The learning rate for the optimizer.

        Returns:
        - None

        """

        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                outputs = self.model(images)  # Forward pass
                loss = self.criterion(outputs, labels)  # Compute the loss
                loss.backward()  # Backward pass
                self.optimizer.step()  # Update the weights
                running_loss += loss.item()
                _, predicted = torch.max(
                    outputs.data, 1
                )  # Get the predicted class with the highest probability
                total += labels.size(0)
                correct += (
                    (predicted == labels).sum().item()
                )  # Number of correct predictions

                # Print metrics at each iteration
                if (i + 1) % 100 == 0:  # Print every 100 batches
                    print(
                        f"Epoch [{epoch + 1}/{self.epochs}], Step [{i + 1}/{len(self.train_loader)}], Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%"
                    )

            # Compute training loss and accuracy
            train_loss = running_loss / len(self.train_loader)
            train_accuracy = 100 * correct / total
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)
            print(
                f"Epoch [{epoch + 1}/{self.epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%"
            )
            # Validate the model
            self.validate()

        # Print out the average training accuracy at the end of training
        avg_train_accuracy = sum(self.train_accuracies) / len(self.train_accuracies)
        print(f"Average Training Accuracy: {avg_train_accuracy:.2f}%")

    def validate(self):
        """
        Computes the validation loss and accuracy of the model. This method is called at the end of each epoch.
        The metrics are stored in the `val_losses` and `val_accuracies` attributes. The
        average validation accuracy is computed at the end of training. The method does not return anything. It
        only prints the metrics.
        The method is called at the end of each epoch, and then the average validation accuracy is
        computed at the end of training.
        """

        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss = running_loss / len(self.test_loader)
        val_accuracy = 100 * correct / total
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy)
        print(
            f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%"
        )
        # Print out the final average validation accuracy
        if len(self.val_accuracies) == self.epochs:
            avg_val_accuracy = sum(self.val_accuracies) / len(self.val_accuracies)
            print(f"Average Validation Accuracy: {avg_val_accuracy:.2f}%")

    def plot_metrics(self):
        """
        Plots the training and validation loss and accuracy.
        """
        epochs = range(1, self.epochs + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label="Training Loss")
        plt.plot(epochs, self.val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss per Epoch")

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, label="Training Accuracy")
        plt.plot(epochs, self.val_accuracies, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy per Epoch")

        plt.show()


# Initialize the model, loss function, and optimizer
model = MyCNN()
trainer = Trainer(model, train_loader, validation_loader)

# Train the model
trainer.train()

# Plot the training and validation metrics
trainer.plot_metrics()

# Save the model
torch.save(model.state_dict(), "best_cnn.pth")
