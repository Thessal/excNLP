from document_model import Trie
import pickle
import numpy as np

class Tokenizer:
    from document_model.explode import explode
    from document_model.explode import assemble
    def __init__(self, working_dir, text_files, rebuild=False, postfix_sensitivity=50):
        """
        Set up tokenizer.
        :param rebuild: If rebuild, read text file and build trie for tokenizer. It takes few hours
        :param text_files: Used for rebuilding
        :param working_dir: Cache files for rebuilding / loading
        :param postfix_sensitivity: postfix detection sensitivity
        """
        self.working_dir = working_dir
        self.trie = Trie()

        cache_files = self._text_to_pkl(files=text_files, chunksize=1000, thres=100, run=rebuild)
        for cache in cache_files:
            with open(cache, 'rb') as handle:
                for c in pickle.load(handle):
                    self.trie.insert(c[0], c[1])

        self.count_all = self.trie.query('')[0][1]
        self.average_postfix_count = postfix_sensitivity

    def _raw_words(self,files):
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
            x = x.replace('/','')
            if not x : pass
            x_e = self.explode(x)

            # Try trie query
            prefix_length_min = 3
            queries = [x_e[:i] for i in range(prefix_length_min,len(x_e)+1)]
            # print(x)
            # print(queries)
            match = [self.trie.query(self.assemble(x)) for x in queries]
            # print([len(x) for x in match])
            match = [x for x in match if x]
            tf = [x[0][1] / self.count_all for x in match]
            # print(tf)
            idf = [-np.log(len(x)/self.average_postfix_count) for x in match]
            # print(idf)
            tfidf = [x[0]*x[1] for x in zip(tf,idf)]
            # print(tfidf)
            prefix_length = prefix_length_min+np.argmax(tfidf) if tfidf else len(x_e)
            # print(prefix_length)
            # print(x_e[:prefix_length])
            # print(x_e[prefix_length:])
            prefix, postfix = self.assemble(x_e[:prefix_length]), self.assemble(x_e[prefix_length:])
            if prefix : output.append(prefix)
            if postfix : output.append(postfix)
        return output