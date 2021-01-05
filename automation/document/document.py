import pandas as pd

class Document:
    def __init__(self, formatter, tokenizer, detokenizer, cache=True):
        self.text = ['']
        self.formatter = formatter
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer
        self._do_cache = cache
        self._cached = False
        self._data = pd.DataFrame()

    def append(self, text='', file=None):
        if file :
            with open(file, 'r') as fp:
                self.text.extend(fp.readlines())
        self.text.extend(text.split('\n'))
        self._cached = False

    def __iter__(self):
        if self._do_cache and (not self._cached):
            data = ((x['begin'][0], x['begin'][1], x['begin'][2], x['text']) for x in
                    self._generate(unit="token", detail=False))
            self._data = pd.DataFrame(columns=["idx_paragraph", "idx_sentence", "idx_token", "token"], data=data)
            self._data.set_index(["idx_paragraph", "idx_sentence", "idx_token"], inplace=True)
            self._cached = True
        if self._do_cache :
            for idx, new_df in self._data.groupby("idx_sentence"):
                tokens = new_df["token"].to_list()
                yield {'sentence': self.detokenizer(tokens), 'tokens': tokens, 'index':new_df[0].index[0]}
        else :
            for sentence in self._generate(unit="tokens", detail=True):
                tokens = [element['text'] for element in sentence['element']]
                yield {'sentence': self.detokenizer(tokens),
                       'tokens': tokens,
                       'index':sentence['begin']}

    def _generate(self, unit="token", detail=False):
        """
        get next item and index
        :param unit: 'token','tokens','sentence','sentences','paragraph','paragraphs','document'
        :return:
        """
        if unit == "sentence": unit = "tokens"
        if unit == "paragraph": unit = "sentences"
        if unit == "document": unit = "paragraphs"
        if unit == "token": detail = False
        if detail: unit = unit + "_detail"
        idx_p = 0
        idx_s = 0
        idx_t = 0
        idxs_p = []
        idxs_s = []
        idxs_t = []
        for p in self.formatter(self.text):  # For each paragraph
            idxs_s.clear()
            idx_tmp_s = (idx_p, idx_s, idx_t)
            for s in p:
                idxs_t.clear()
                idx_tmp_w = (idx_p, idx_s, idx_t)
                for t in self.tokenizer(s):
                    if unit == "token": yield {'text': t, 'begin': (idx_p, idx_s, idx_t)}
                    if unit == "tokens_detail": idxs_t.append((idx_p, idx_s, idx_t))
                    idx_t += 1
                if unit == "tokens": yield {'text': s, 'begin': idx_tmp_w, 'end': (idx_p, idx_s, idx_t)}
                if unit == "tokens_detail": yield {'element': [{'text': x, 'begin': y} for x, y in zip(s, idxs_t)],
                                                   'begin': idx_tmp_w,
                                                   'end': (idx_p, idx_s, idx_t)}
                if unit == "sentences_detail": idxs_s.append(idx_tmp_w)
                idx_s += 1
            if unit == "sentences": yield {'text': p, 'begin': idx_tmp_s, 'end': (idx_p, idx_s, idx_t)}
            if unit == "sentences_detail": yield {'element': [{'text': x, 'begin': y} for x, y in zip(s, idxs_s)],
                                                  'begin': idx_tmp_s,
                                                  'end': (idx_p, idx_s, idx_t)}
            if unit == "paragraphs_detail": idxs_p.append(idx_tmp_s)
            idx_p += 1
