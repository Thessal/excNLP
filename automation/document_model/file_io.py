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

    _documents = lambda paragraphs: ' '.join([x.strip() for x in paragraphs if x.strip()])
    _sentences = lambda paragraphs: [sentences.split(". ") for sentences in paragraphs]

    def documents(self):
        for p in self.paragraphs():
            yield self._documents.__func__(p)

    def sentences(self):  # Approximate
        for p in self.paragraphs():
            yield self._sentences.__func__(p)

    def all(self):
        for p in self.paragraphs():
            yield {"paragraphs":p,
                   "documents":self._documents.__func__(p),
                   "sentences":self._sentences.__func__(p)}
