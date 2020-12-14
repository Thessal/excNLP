# Document statistics

class TextIO:
    def __init__(self, pattern):
        self.pattern = pattern

    def files(self):
        import glob
        for file in list(glob.glob(self.pattern)):
            with open(file, "r", encoding="utf-8") as f:
                yield f

    def paragraphs(self):
        for fp in self.files():
            yield fp.readlines()

    def documents(self):
        for p in self.paragraphs():
            yield ' '.join([x.strip() for x in p if x.strip()])

    def sentences(self):  # Approximate
        for d in self.documents():
            yield d.split(". ")


from trie import *


class DocStat(TextIO):
    def __init__(self, pattern):
        super().__init__(pattern)

    def _words(self, long_string):
        return long_string.split(' ')

    def _trie(self, list_words):
        t = Trie()
        for w in (explode(w) for w in list_words):
            t.insert(w)
        return t

    def trie(self):
        for d in self.documents():
            yield self._trie(self._words(d))


path = "proprietary/text/*.txt"
doc_stat = DocStat(path)
for x in doc_stat.trie():
    search = {assemble(w[0]): w[1] for w in x.query(explode('승'))}
    print(x.top_n_items(30))
    # TODO : do TF-IDF to find keywords
    break
