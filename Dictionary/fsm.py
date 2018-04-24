import json
from nltk import word_tokenize

class FSM(object):
    """
    FSM class with state machine functionality.
    """

    def __init__(self):
        self.state_dictionary = {}

    def add_document(self, path, data="train"):
        """
        Adds all the words in a document to the FSM.
        States in the FSM represent words and transitions represent sequential
        ordering of words
        """
        parsed_document = json.load(open(path, 'r'))

        if "sections" not in parsed_document:
            return

        sections = parsed_document["sections"]

        # parse words from every section of the paper except for References
        exclude = ["References"]
        for section in sections:
            if "heading" in section and section["heading"] not in exclude:
                words = word_tokenize(section["text"])
                for i in range(len(words) - 1):
                    self.add_transition(words[i], words[i+1])

    def add_transition(self, s1, s2):
        if s1 not in self.state_dictionary:
            self.state_dictionary[s1] = set()
        self.state_dictionary[s1].add(s2)

    def delete_states(self, on_states, grammar_states):
        to_delete = []
        for word in self.state_dictionary:
            if word not in on_states or word not in grammar_states:
                to_delete.append(word)
        for word in to_delete:
            del self.state_dictionary[word]
