import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import warnings

warnings.filterwarnings("ignore")


def flatten_pad_sequences(input_tensor: torch.tensor, mask: torch.tensor):
    """
    Flat paded sequences, removing the padding.
    """

    packed_input = pack_padded_sequence(
        input_tensor,
        mask.sum(1).to(torch.int64),
        batch_first=True,
        enforce_sorted=False,
    )

    return packed_input.data


class Segmentator(nn.Module):
    def __init__(
        self, n_classes: int, input_size: int, lstm_hidden_size: int = 256,
    ):
        super(Segmentator, self).__init__()

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=lstm_hidden_size, batch_first=True,
        )
        self.lstm_top = nn.Linear(lstm_hidden_size, n_classes,)

    def forward(self, input_features, mask):
        features_packed = pack_padded_sequence(
            input_features,
            mask.sum(1).to(torch.int64),
            batch_first=True,
            enforce_sorted=False,
        )

        x, hidden_states = self.lstm(features_packed)

        # Flatten data with all outputs in the same dimension
        # Ex: 120x256
        output = self.lstm_top(x.data)

        # Repack the output to generate each sequence
        packed_output = torch.nn.utils.rnn.PackedSequence(
            output, x.batch_sizes, x.sorted_indices, x.unsorted_indices,
        )

        # Pad and return solution
        paded_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        return paded_output
