import json
from nltk import word_tokenize

"""
TODO:
    - Hardcode grammar states
    - Opt for online exclusion instead of complete deletion
"""


class FSM(object):
    """
    FSM class with state machine functionality.
    """

    def __init__(self):
        self.state_dictionary = {}
        self.grammar_states = set()
        self.allowable_states = set()

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

    def set_on_states(self, on_states):
        # Legal transitions in the FSM will only be between states
        # in the union of the on-states and grammar-states.
        self.allowable_states = on_states + self.grammar_states

    def get_neighbors(self, word):
        candidate_neighbors = self.state_dictionary[word]
        filtered_neighbors = [c for c in candidate_neighbors
                              if c in self.allowable_states]

        return filtered_neighbors
