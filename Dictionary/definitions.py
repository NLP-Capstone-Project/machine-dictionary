class UMLS(object):
    """
    UMLS class with definition parsing functionality.
    example usage:
    obj = UMLS()
    test_path = "/data-collection/UMLS_samples/MRCONSO.RRF"
    obj.parse_synonym_file(test_path)
    """

    def __init__(self):
        self.id2definition = {}
        self.term2id = {}

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
        for line in file:
            split = line.split('|')
            id = split[0]
            term = split[14]
            self.term2id[term] = id
