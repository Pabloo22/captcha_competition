import torch


class CustomAccuracyMetric:
    def __init__(self):
        self.correct_predictions = 0
        self.total_predictions = 0

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update the metric with new predictions and targets.

        Parameters:
        - outputs: torch.Tensor of shape (batch_size, 10, 6) - model predictions.
        - targets: torch.Tensor of shape (batch_size, 6) - true labels.
        """
        with torch.no_grad():
            predicted_classes = outputs.argmax(
                dim=1
            )  # Get the predicted classes for each column
            correct = (predicted_classes == targets).sum(
                dim=1
            )  # Sum correct predictions per sample
            self.correct_predictions += (
                correct.sum().item()
            )  # Update total correct predictions
            self.total_predictions += (
                targets.numel()
            )  # Update total number of predictions

    def compute(self) -> float:
        """
        Compute the accuracy metric.

        Returns:
        - The accuracy as a float.
        """
        if self.total_predictions == 0:
            return 0.0  # Avoid division by zero
        accuracy = self.correct_predictions / self.total_predictions
        return accuracy

    def reset(self) -> None:
        """
        Reset the metric's internal state.
        """
        self.correct_predictions = 0
        self.total_predictions = 0
