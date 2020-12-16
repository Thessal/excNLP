# Document can be large. Write files rather than using in-memory document.

class TextIO:
    def __init__(self, pattern):
        self.pattern = pattern

    def files(self):
        import glob
        for file in list(glob.glob(self.pattern)):
            with open(file, "r", encoding="utf-8") as f:
                yield f

    def paragraphs(self):  # lines, actually
        for fp in self.files():
            yield [x.strip() for x in fp.readlines() if x.strip()]

    _documents = lambda paragraphs: ' '.join([x.strip() for x in paragraphs if x.strip()])
    _sentences = lambda paragraphs: (sentences.split(". ") for sentences in paragraphs)  # Approximate
    _words = lambda sentences: (sentence.split(' ') for sentence in sentences if sentence.split(' '))  # Approximate

    def documents(self):
        for p in self.paragraphs():
            yield self._documents.__func__(p)

    def sentences(self):
        for p in self.paragraphs():
            yield self._sentences.__func__(p)

    def words(self):
        for s in self.sentences():
            yield self._words.__func__(s)

    document = sentences
    sentence = words

    def all(self):
        for p in self.paragraphs():
            yield {"paragraphs": p,
                   "documents": self._documents.__func__(p),
                   "sentences": self._sentences.__func__(p)}

    # FIXME : refactor to use dict rather than list
    def generate(self, unit="word", detail=False):
        """
        get next item and index
        :param unit: 'word','words','sentence','sentences','paragraph','paragraphs','document'
        :return:
        """
        if unit == "sentence": unit = "words"
        if unit == "paragraph": unit = "sentences"
        if unit == "document": unit = "paragraphs"
        if unit == "word": detail = False
        if detail: unit = unit + "_detail"
        idx_d = 0
        idx_p = 0
        idx_s = 0
        idx_w = 0
        idxs_p = []
        idxs_s = []
        idxs_w = []
        for ps in self.paragraphs(): # For each document
            idxs_p.clear()
            idx_tmp_p = (idx_d, idx_p, idx_s, idx_w)
            for ss in self._sentences.__func__(ps): # For each paragraph
                idxs_s.clear()
                idx_tmp_s = (idx_d, idx_p, idx_s, idx_w)
                for ws in self._words.__func__(ss):
                    idxs_w.clear()
                    idx_tmp_w = (idx_d, idx_p, idx_s, idx_w)
                    for w in ws:
                        if unit == "word": yield {'text': w, 'begin': (idx_d, idx_p, idx_s, idx_w)}
                        if unit == "words_detail": idxs_w.append((idx_d, idx_p, idx_s, idx_w))
                        idx_w += 1
                    if unit == "words": yield {'text': ws, 'begin': idx_tmp_w, 'end': (idx_d, idx_p, idx_s, idx_w)}
                    if unit == "words_detail": yield {'element': [{'text':x, 'begin':y} for x,y in zip(ws, idxs_w)], 'begin': idx_tmp_w,
                                                      'end': (idx_d, idx_p, idx_s, idx_w)}
                    if unit == "sentences_detail": idxs_s.append(idx_tmp_w)
                    idx_s += 1
                if unit == "sentences": yield {'text': ss, 'begin': idx_tmp_s, 'end': (idx_d, idx_p, idx_s, idx_w)}
                if unit == "sentences_detail": yield {'element': [{'text':x, 'begin':y} for x,y in zip(ss, idxs_s)], 'begin': idx_tmp_s,
                                                      'end': (idx_d, idx_p, idx_s, idx_w)}
                if unit == "paragraphs_detail": idxs_p.append(idx_tmp_s)
                idx_p += 1
            if unit == "paragraphs": yield {'text': ps, 'begin': idx_tmp_p, 'end': (idx_d, idx_p, idx_s, idx_w)}
            if unit == "paragraphs_detail": yield {'element': [{'text':x, 'begin':y} for x,y in zip(ps, idxs_p)], 'begin': idx_tmp_p,
                                                   'end': (idx_d, idx_p, idx_s, idx_w)}
            idx_d += 1
