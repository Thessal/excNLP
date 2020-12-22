# https://albertauyeung.github.io/2020/06/15/python-trie.html

class TrieNode:
    """A node in the trie structure"""

    def __init__(self, char):
        # the character stored in this node
        self.char = char

        # whether this can be the end of a word
        self.is_end = False

        # a counter indicating how many times a word is inserted
        # (if this node's is_end is True)
        self.counter = 0
        self.counter_recursive = 0

        # a dictionary of child nodes
        # keys are characters, values are nodes
        self.children = {}


class Trie(object):
    from .explode import assemble as _assemble
    from .explode import explode as _explode
    """The trie object"""

    def __init__(self, list_words=[], auto_explode=True):
        """
        The trie has at least the root node.
        The root node does not store any character
        """
        self.root = TrieNode("")
        self.output = []
        self.assemble = self._assemble if auto_explode else lambda x: x
        self.explode = self._explode if auto_explode else lambda x: x
        for w in (w for w in list_words):
            self.insert(w)

    def insert(self, word):
        """Insert a word into the trie"""
        word = self.explode(word)
        if len(word)>30 : return
        node = self.root

        # Loop through each character in the word
        # Check if there is no child containing the character, create a new child for the current node
        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                # If a character is not found,
                # create a new node in the trie
                new_node = TrieNode(char)
                node.children[char] = new_node
                node = new_node
            node.counter_recursive += 1

        # Mark the end of a word
        node.is_end = True

        # Increment the counter to indicate that we see this word once more
        node.counter += 1

    def dfs(self, node, prefix, include_subword=True):
        """Depth-first traversal of the trie

        Args:
            - node: the node to start with
            - prefix: the current prefix, for tracing a
                word while traversing the trie
        """
        if include_subword:
            if node.is_end or len(node.children) > 2:
                self.output.append((self.assemble(prefix + node.char), node.counter_recursive))

        else:
            if node.is_end:
                self.output.append((self.assemble(prefix + node.char), node.counter))

        for child in node.children.values():
            self.dfs(child, prefix + node.char)

    def query(self, x):
        """Given an input (a prefix), retrieve all words stored in
        the trie with that prefix, sort the words by the number of
        times they have been inserted
        """
        x = self.explode(x)  # .__call__(x)
        # Use a variable within the class to keep all possible outputs
        # As there can be more than one word with such prefix
        self.output = []
        node = self.root

        # Check if the prefix is in the trie
        for char in x:
            if char in node.children:
                node = node.children[char]
            else:
                # cannot found the prefix, return empty list
                return []

        # Traverse the trie to get all candidates
        self.dfs(node, x[:-1])

        # Sort the results in reverse order and return
        return sorted(self.output, key=lambda x: x[1], reverse=True)

    def include_hangul(k):
        return any([(ord('가') <= ord(x) <= ord('힣')) for x in k])
        # 자음 모음 : (12593 <= ord(x) <= 12686)
        #'힣' = '가' + 19*21*28 -1

    def top_n_items(self, n):
        freq = {w[0]: w[1] for w in self.query('')}
        freq = {k: v for k, v in freq.items() if (len(k) > 1) and self.include_hangul.__func__(k)}
        top_words = sorted(freq.keys(), key=freq.get, reverse=True)[:min(n, len(freq))]
        return {w: freq[w] for w in top_words}

# https://towardsdatascience.com/implementing-a-trie-data-structure-in-python-in-less-than-100-lines-of-code-a877ea23c1a1
# from typing import Tuple
#
#
# class TrieNode(object):
#     """
#     Our trie node implementation. Very basic. but does the job
#     """
#
#     def __init__(self, char: str):
#         self.char = char
#         self.children = []
#         # Is it the last character of the word.`
#         self.word_finished = False
#         # How many times this character appeared in the addition process
#         self.counter = 1
#
#
# def add(root, word: str):
#     """
#     Adding a word in the trie structure
#     """
#     node = root
#     for char in word:
#         found_in_child = False
#         # Search for the character in the children of the present `node`
#         for child in node.children:
#             if child.char == char:
#                 # We found it, increase the counter by 1 to keep track that another
#                 # word has it as well
#                 child.counter += 1
#                 # And point the node to the child that contains this char
#                 node = child
#                 found_in_child = True
#                 break
#         # We did not find it so add a new chlid
#         if not found_in_child:
#             new_node = TrieNode(char)
#             node.children.append(new_node)
#             # And then point node to the new child
#             node = new_node
#     # Everything finished. Mark it as the end of a word.
#     node.word_finished = True
#
#
# def find_prefix(root, prefix: str) -> Tuple[bool, int]:
#     """
#     Check and return
#       1. If the prefix exsists in any of the words we added so far
#       2. If yes then how may words actually have the prefix
#     """
#     node = root
#     # If the root node has no children, then return False.
#     # Because it means we are trying to search in an empty trie
#     if not root.children:
#         return False, 0
#     for char in prefix:
#         char_not_found = True
#         # Search through all the children of the present `node`
#         for child in node.children:
#             if child.char == char:
#                 # We found the char existing in the child.
#                 char_not_found = False
#                 # Assign node as the child containing the char and break
#                 node = child
#                 break
#         # Return False anyway when we did not find a char.
#         if char_not_found:
#             return False, 0
#     # Well, we are here means we have found the prefix. Return true to indicate that
#     # And also the counter of the last node. This indicates how many words have this
#     # prefix
#     return True, node.counter
#
#
# if __name__ == "__main__":
#     root = TrieNode('*')
#     add(root, "hackathon")
#     add(root, 'hack')
#
#     print(find_prefix(root, 'hac'))
#     print(find_prefix(root, 'hack'))
#     print(find_prefix(root, 'hackathon'))
#     print(find_prefix(root, 'ha'))
#     print(find_prefix(root, 'hammer'))
