import torch
import torch.nn.functional as F


class CustomCategoricalCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        # inputs shape: (batch_size, 10, 6) or (batch_size, 6, 10)
        # targets shape: (batch_size, 6)

        _, num_rows, num_columns = inputs.shape
        losses = []

        for i in range(6):
            # Slice the predictions and labels for each column
            if num_columns == 6:
                predictions = inputs[:, :, i]
            elif num_rows == 6:
                predictions = inputs[:, i, :]
            else:
                raise ValueError(f"Invalid input shape: {inputs.shape}")
            target_vectors = targets[:, i]

            # Apply categorical cross-entropy for each column
            loss = F.cross_entropy(predictions, target_vectors)
            losses.append(loss)

        # Calculate the mean loss across all columns
        mean_loss = torch.mean(torch.stack(losses))

        return mean_loss
