

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