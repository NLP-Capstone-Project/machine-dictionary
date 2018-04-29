
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.functional import cross_entropy

from Dictionary.utils import word_vector_from_seq

from metrics.feature_extraction.definition_classifier \
    import DefinitionClassifier


def evaluate_representations(model, corpus, batch_size, bptt_limit, cuda):
    """
    Measure the richness of the hidden states by using them as features
    for a multi-class regression model.

    Assumes that each section is a passage with an omitted-term and a
    'target' to predict.

    We first train a two-layer neural network to serve as the classifier
    before evaluation begins.
    """

    classifier_data = []
    feature_size = None
    target_set = set()

    # Collect the data for the classifier (harvest hidden states and pair
    # with targets).
    model.eval()
    print("Evaluation in progress: Representations")
    for i, document in enumerate(corpus.test):

        for j, section in enumerate(document["sections"]):
            hidden = model.init_hidden()

            # Training at the word level allows flexibility in inference.
            for k in range(section.size(0) - 1):
                current_word = word_vector_from_seq(section, k)

                if cuda:
                    current_word = current_word.cuda()

                output, hidden = model(Variable(current_word), hidden)

            # Flatten the model's final hidden states to be used as features.
            features = hidden.view(-1, 1)
            if feature_size is None:
                feature_size = features.size()[0]

            # Each section has a corresponding target.
            target = document["targets"][j]
            target_set.add(target)
            example = {
                "features": features.data,
                "target": target
            }

            classifier_data.append(example)

    # See how well the hidden states represent semantics:
    target_size = len(target_set)
    definition_classifier = DefinitionClassifier(feature_size, 128, target_size)
    optimizer = torch.optim.Adam(definition_classifier.parameters(), lr=0.005)
    epochs = 2
    classifier_accuracy = train_classifier(definition_classifier, classifier_data,
                                           epochs, target_set, optimizer, cuda)

    print("Classification accuracy from hidden state features: {.5f}"
          .format(classifier_accuracy))


def train_classifier(model, data, epochs, target_set, optimizer, cuda):
    """
    Train a definition classifier for one epoch.
    """

    total_loss = 0

    # Establish labels for each target:
    target_mappings = {}
    target_list = list(target_set)
    for i, target in enumerate(target_list):
        target_mappings[target] = i

    # Helper function for conversion to Variable for learning.
    def target_to_variable(target):
        target_variable = torch.LongTensor(1)
        target_variable[0] = target_mappings[target]
        target_variable = Variable(target_variable)
        return target_variable

    # Set model to training mode (activates dropout and other things).
    model.train()
    print("Classifier Training in progress:")
    for _ in range(epochs):
        for i, example in enumerate(data):
            features = example["features"]
            target = target_to_variable(example["target"])

            if cuda:
                features = features.cuda()
                target = target.cuda()

            output = model(Variable(features))

            # Calculate loss between the next word and what was anticipated.
            loss = cross_entropy(output.view(1, -1), target)
            total_loss += loss.data[0]

            # Backpropagation.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print("Total loss from training: {.5f}".format(total_loss[0]))

    # Calculate final accuracy.
    model.eval()
    correct = 0
    for i, example in enumerate(data):
        features = example["features"]
        target = target_mappings[example["target"]]

        if cuda:
            features = features.cuda()
            target = target.cuda()

        output = model(Variable(features))
        _, prediction = torch.max(output, 0)

        if prediction == target:
            correct += 1

    final_acc = correct / len(data)

    return final_acc
