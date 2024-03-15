import torch
import torch.nn.functional as F


class CustomCategoricalCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        # inputs shape: (batch_size, 10, 6) - predictions
        # targets shape: (batch_size, 6) - true labels

        _, _, num_columns = inputs.shape
        losses = []

        for i in range(num_columns):
            # Slice the predictions and labels for each column
            predictions_column = inputs[:, :, i]
            targets_column = targets[:, i]

            # Apply categorical cross-entropy for each column
            loss = F.cross_entropy(predictions_column, targets_column)
            losses.append(loss)

        # Calculate the mean loss across all columns
        mean_loss = torch.mean(torch.stack(losses))

        return mean_loss
