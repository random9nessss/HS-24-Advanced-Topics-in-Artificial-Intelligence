class PrefixNode:
    """
    A node in the Prefix Tree (Trie) data structure.

    Attributes:
        children (dict): A dictionary mapping tokens to child PrefixNodes.
        is_end_of_entity (bool): Indicates if the node represents the end of an entity.
        original_entity (str): The original entity string stored at the end node.
    """

    def __init__(self):
        self.children = {}
        self.is_end_of_entity = False
        self.original_entity = None

class PrefixTree:
    """
    A Prefix Tree (Trie) for efficient matching of entities within text.

    Attributes:
        root (PrefixNode): The root node of the Prefix Tree.
    """

    def __init__(self):
        self.root = PrefixNode()

    def insert(self, entity_tokens, original_entity):
        """
        Insert an entity into the Prefix Tree.

        Args:
            entity_tokens (list of str): A list of tokens representing the entity.
            original_entity (str): The original entity string.
        """
        node = self.root
        for token in entity_tokens:
            if token not in node.children:
                node.children[token] = PrefixNode()
            node = node.children[token]
        node.is_end_of_entity = True
        node.original_entity = original_entity

    def search(self, tokens, start_index):
        """
        Search for the longest matching entity starting from a given index in the tokens list.

        Args:
            tokens (list of str): The list of tokens to search within.
            start_index (int): The index to start searching from.

        Returns:
            tuple: A tuple containing the matched entity (or None if no match)
                   and the index where the match ends.
        """
        node = self.root
        current_index = start_index
        last_matching_index = -1
        last_matching_entity = None

        while current_index < len(tokens):
            token = tokens[current_index]
            if token in node.children:
                node = node.children[token]
                if node.is_end_of_entity:
                    last_matching_index = current_index
                    last_matching_entity = node.original_entity
                current_index += 1
            else:
                break

        if last_matching_index != -1:
            return last_matching_entity, last_matching_index
        else:
            return None, start_index