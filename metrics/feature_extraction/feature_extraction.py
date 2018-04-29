
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.functional import cross_entropy


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
