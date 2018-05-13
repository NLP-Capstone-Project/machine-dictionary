
class UMLS(object):
    """
    UMLS class with definition parsing functionality.
    example usage:
    obj = UMLS()
    test_path = "/data-collection/UMLS_samples/MRCONSO.RRF"
    obj.parse_synonym_file(test_path)
    """

    def __init__(self, def_path, syn_path):
        self.id2definition = {}
        self.term2id = {}
        self.def_path = def_path
        self.syn_path = syn_path
        self.terms = []
        self.definition_mappings = {}

    def parse_definition_file(self, path):
        """
        Parses the UMLS file MRDEF.RRF, which contains term ids and their definitions
        """
        file = open(path, 'r')
        next(file)
        for line in file:
            split = line.split('|')
            id = split[0]
            definition = split[5]
            self.id2definition[id] = definition

    def parse_synonym_file(self, path):
        """
        Parses the UMLS file MRCONSO.RRF, which contains term ids and their synonyms
        """
        file = open(path, 'r')
        next(file)
        for line in file:
            split = line.split('|')
            id = split[0]
            term = split[14]
            self.term2id[term] = id

    def generate_all_definitions(self):
        self.parse_definition_file(self.def_path)
        self.parse_synonym_file(self.syn_path)

        for term in self.term2id:
            if self.term2id[term] in self.id2definition:
                definition = self.id2definition[self.term2id[term]]
                definition_object = {
                    "term": term,
                    "definition": definition
                }
                self.terms.append(definition_object)

        # Map definitions to all the terms it defines.
        for term in self.terms:
            definition = term["definition"]
            if definition not in self.definition_mappings:
                self.definition_mappings[definition] = set()

            self.definition_mappings[definition].add(term["term"])
