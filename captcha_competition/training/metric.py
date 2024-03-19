import torch


class CustomAccuracyMetric:
    def __init__(self, per_digit: bool = True):
        self.correct_predictions = 0
        self.total_predictions = 0
        self.per_digit = per_digit

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update the metric with new predictions and targets.

        Parameters:
        - outputs: torch.Tensor of shape (batch_size, 10, 6) - model predictions.
        - targets: torch.Tensor of shape (batch_size, 6) - true labels.
        """
        with torch.no_grad():
            predicted_classes = self.get_predicted_classes(outputs)
            correct = (predicted_classes == targets).sum(dim=1)

            if self.per_digit:
                self.correct_predictions += correct.sum().item()  # type: ignore
                self.total_predictions += targets.numel()
            else:
                self.correct_predictions += (correct == 6).sum().item()  # type: ignore
                self.total_predictions += targets.size(0)

    @staticmethod
    def get_predicted_classes(outputs: torch.Tensor) -> torch.Tensor:
        # If the model is a ResNetTransformer, the shape of the outputs tensor is
        # (batch_size, 6, 10)
        # Otherwise, the shape is (batch_size, 10, 6)
        if outputs.shape[1] == 6:
            predicted_classes = outputs.argmax(dim=2)
        elif outputs.shape[2] == 6:
            predicted_classes = outputs.argmax(dim=1)
        else:
            raise ValueError(f"Invalid input shape: {outputs.shape}")
        return predicted_classes

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


def custom_accuracy(
    outputs: torch.Tensor, targets: torch.Tensor, per_digit: bool = True
) -> float:
    """
    Compute the accuracy metric.

    Parameters:
    - outputs: torch.Tensor of shape (batch_size, 10, 6) - model predictions.
    - targets: torch.Tensor of shape (batch_size, 6) - true labels.

    Returns:
    - The accuracy as a float.
    """
    accuracy_metric = CustomAccuracyMetric(per_digit)
    accuracy_metric.update(outputs, targets)
    return accuracy_metric.compute()
