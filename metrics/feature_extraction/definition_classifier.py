from torch.autograd import Variable
import torch.nn as nn


class DefinitionClassifier(nn.Module):

    def __init__(self, feature_size, hidden_size, target_size,
                 dropout=0.5):

        """
        A two-layer neural network that accepts hidden states to predict the
        omitted target term from a passage missing the target term.

        The purpose of this model is to determine if the latent representations
        of the language model are rich enough to indicative of the words they
        are encoding.

        We allow the model to be minimal to test the richness of the features.

        Parameters:
        -----------
        :param feature_size: int
            The number of features we've selected from the model.
        :param target_size: int
            The size of the vocabulary in which to predict.
        :param hidden_size: int
            The number of hidden dimensions to map the input to before
            the output layer.
        """
        # Save the construction arguments, useful for serialization
        self.init_arguments = locals()
        self.init_arguments.pop("self")
        self.init_arguments.pop("__class__")
        super(DefinitionClassifier, self).__init__()

        self.target_size = target_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.model = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_size, target_size)
        )

    def forward(self, input):

        # Forward pass.
        # Shape: (1, target_size)
        return self.model(input)
