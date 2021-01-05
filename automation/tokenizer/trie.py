
# from util.explode import explode
# from util.explode import assemble
# def __init__(self, working_dir, text_files, rebuild=False, postfix_sensitivity=50):
#     """
#     Set up tokenizer.
#     :param rebuild: If rebuild, read text file and build trie for tokenizer. It takes few hours
#     :param text_files: Used for rebuilding
#     :param working_dir: Cache files for rebuilding / loading
#     :param postfix_sensitivity: postfix detection sensitivity
#     """
#     self.working_dir = working_dir
#     self.trie = Trie()
#
#     cache_files = self._text_to_pkl(files=text_files, chunksize=1000, thres=100, run=rebuild)
#     for cache in cache_files:
#         with open(cache, 'rb') as handle:
#             for c in pickle.load(handle):
#                 self.trie.insert(c[0], c[1])
#
#     self.count_all = self.trie.query('')[0][1]
#     self.average_postfix_count = postfix_sensitivity

def _raw_words(self, files):
    """
    Generates words from files
    :param files: list containing txt filenames
    :return: word generator
    """
    for file in files:
        with open(file, "r", encoding="utf-8") as fp:
            for line in fp.readlines():
                if not line.strip(): pass
                words = line.strip().split(' ')
                for word in words:
                    word = word.strip()
                    if word: yield word
    print(f"{len(files)} files processed.")

def _text_to_pkl(self, files, chunksize=1000, thres=100, run=False):
    """
    In one loop, read {chunksize} files.
    For each loop, truncate words not popular than {thres}.
    :param files: text files
    :param chunksize: generate a trie for every {chunksize} files.
    :param thres: for each trie, truncate words not popular than {thres}.
    :param run: if False, then only generate pkl paths
    :return: pkl file path
    """
    print(f"Total {len(files)} files.")
    indices = list(range(int(len(files) / chunksize)))
    pkl_files = [f"{self.working_dir}Trie_{x}.pkl" for x in indices]
    if run:
        for x in indices:
            begin = x * chunksize
            end = min(len(files), begin + chunksize)
            trie = Trie()
            for word in self._raw_words(files=files[begin:end]):
                trie.insert(word)
            trie.output = []
            trie.dfs(trie.root, '', include_subword=True, count_recursive=False)
            data = [x for x in trie.output if x[1] > thres]
            with open(pkl_files[x], 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return pkl_files

def tokenize(self, text):
    """
    Tokenize input text
    :param text:
    :return:
    """
    output = []
    for x in text.strip().split(' '):
        x = x.replace('/', '')
        if not x: pass
        x_e = self.explode(x)

        # Try trie query
        prefix_length_min = 3
        queries = [x_e[:i] for i in range(prefix_length_min, len(x_e) + 1)]
        # print(x)
        # print(queries)
        match = [self.trie.query(self.assemble(x)) for x in queries]
        # print([len(x) for x in match])
        match = [x for x in match if x]
        tf = [x[0][1] / self.count_all for x in match]
        # print(tf)
        idf = [-np.log(len(x) / self.average_postfix_count) for x in match]
        # print(idf)
        tfidf = [x[0] * x[1] for x in zip(tf, idf)]
        # print(tfidf)
        prefix_length = prefix_length_min + np.argmax(tfidf) if tfidf else len(x_e)
        # print(prefix_length)
        # print(x_e[:prefix_length])
        # print(x_e[prefix_length:])
        prefix, postfix = self.assemble(x_e[:prefix_length]), self.assemble(x_e[prefix_length:])
        if prefix: output.append(prefix)
        if postfix: output.append(postfix)
    return output




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
    from util.explode import assemble as _assemble
    from util.explode import explode as _explode
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

    def insert(self, word, count=1):
        """
        Insert a word into the trie
        :param word: word to add
        :param count: add count times
        """
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
            node.counter_recursive += count

        # Mark the end of a word
        node.is_end = True

        # Increment the counter to indicate that we see this word once more
        node.counter += count

    def validate(self):
        """
        Validate counter and counter_recursive
        :return:
        """
        raise NotImplementedError

    def dfs(self, node, prefix, include_subword=True, count_recursive=True):
        """Depth-first traversal of the trie

        Args:
            - node: the node to start with
            - prefix: the current prefix, for tracing a
                word while traversing the trie
            - include_subword : if false, only return leaves. if true,
                 popular subword nodes are also returned (childeren>=3)
            - count_recursive : if true, count self and child recursively
        """
        # NOTE : unpopular nodes are always truncated...
        if node.is_end or (include_subword and (len(node.children) > 2)):
            self.output.append((
                self.assemble(prefix + node.char),
                (node.counter_recursive if count_recursive else node.counter)
            ))

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
