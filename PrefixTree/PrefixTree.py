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

    def search_approximate(self, tokens, start_index, max_edits=1):
        """
        Search for the longest matching entity starting from a given index in the tokens list,
        allowing for up to max_edits typos only if the total string length exceeds 5 characters.

        Args:
            tokens (list of str): The list of tokens to search within.
            start_index (int): The index to start searching from.
            max_edits (int): Maximum number of allowed typos.

        Returns:
            tuple: A tuple containing the matched entity (or None if no match)
                   and the index where the match ends.
        """
        results = []

        def recursive_search(node, current_index, edits_made, total_length):
            if edits_made > max_edits:
                return
            if node.is_end_of_entity:
                results.append((node.original_entity, current_index - 1))
            if current_index >= len(tokens):
                return

            token = tokens[current_index]
            for child_token, child_node in node.children.items():
                new_total_length = total_length + len(child_token)

                # Exact match
                if child_token == token:
                    recursive_search(child_node, current_index + 1, edits_made, new_total_length)

                # Approximate match allowed only if total length > 5
                elif new_total_length > 5 and is_within_edit_distance_one(token, child_token):
                    recursive_search(child_node, current_index + 1, edits_made + 1, new_total_length)

        recursive_search(self.root, start_index, 0, 0)

        if results:
            return max(results, key=lambda x: (x[1], -start_index))

        else:
            return None, start_index

def is_within_edit_distance_one(s1, s2):
    """
    Check if two strings are within an edit distance of 1.

    Args:
        s1 (str): First string.
        s2 (str): Second string.

    Returns:
        bool: True if the edit distance between s1 and s2 is less than or equal to 1.
    """
    if s1 == s2:
        return True
    len_s1, len_s2 = len(s1), len(s2)
    if abs(len_s1 - len_s2) > 1:
        return False
    edits = 0
    i = j = 0
    while i < len_s1 and j < len_s2:
        if s1[i] == s2[j]:
            i += 1
            j += 1
        else:
            edits += 1
            if edits > 1:
                return False
            if len_s1 == len_s2:
                i += 1
                j += 1
            elif len_s1 > len_s2:
                i += 1
            else:
                j += 1

    if i < len_s1 or j < len_s2:
        edits += 1
    return edits <= 1