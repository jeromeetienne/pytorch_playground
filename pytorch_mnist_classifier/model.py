# Importing dependencies
from torch import nn
import torch


# Define the image classifier model
class ImageClassifierModel(nn.Module):
    """
    A simple convolutional neural network for classifying MNIST images.
    The model consists of three convolutional layers followed by a fully connected layer.
    """
    def __init__(self):
        super(ImageClassifierModel, self).__init__()
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
        )
        self.fully_connected_layers = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 22 * 22, 10)
        )

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = self.fully_connected_layers(x)
        return x

    @staticmethod
    def do_train(
        model: "ImageClassifierModel",
        train_dataloader: torch.utils.data.DataLoader,
        epochs=10,
        log_enabled=False,
        device=None,
    ):
        """
        Train the image classifier model on the training dataset.

        Args:
            model (ImageClassifierModel): The model to train.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            epochs (int, optional): Number of epochs to train the model. Defaults to 10.
            log_enabled (bool, optional): Whether to log training progress. Defaults to False.
            device (str, optional): Device to run the training on. Defaults to None.
        """

        # Define the optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
        loss_fn = nn.CrossEntropyLoss()

        # guess the device if not provided
        if device is None:
            device = (
                torch.accelerator.current_accelerator().type
                if torch.accelerator.is_available()
                else "cpu"
            )

        model.to(device)
        model.train()  # Set the model to training mode

        for epoch in range(epochs):
            for images, labels in train_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

            if log_enabled:
                print(f"Epoch:{epoch} - loss is {(loss.item()*100.0):.4f}%")

    @staticmethod
    def do_eval(
        model: "ImageClassifierModel",
        test_dataloader: torch.utils.data.DataLoader,
        device=None,
    ) -> float:
        """
        Evaluate the model on the test dataset.

        Args:
            model (ImageClassifierModel): The trained model to evaluate.
            test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
            device (str, optional): Device to run the evaluation on. Defaults to None.
        
        Returns:
            float: The accuracy of the model on the test dataset as a percentage.
        """

        # guess the device if not provided
        if device is None:
            device = (
                torch.accelerator.current_accelerator().type
                if torch.accelerator.is_available()
                else "cpu"
            )

        model.to(device)
        model.eval()  # Set the model to evaluation mode

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)  # Forward pass

                _, predicted = torch.max(
                    outputs, dim=1
                )
                
                # Get predicted class from output probabilities
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy

