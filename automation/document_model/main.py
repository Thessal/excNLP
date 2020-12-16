from .file_io import TextIO
import os
from .trie import Trie

## Tokenizer endcoder related
import pickle
import glob
from konlpy.tag import Kkma


class Document(TextIO):
    """
    Generate results for each files
    """

    def __init__(self, pattern, limit=None):
        """
        :param pattern: "/foo/bar/baz/wildcard.txt"
        :param limit: if specified, limits max line
        """
        super().__init__(pattern)
        replace_extension = lambda file, ext: os.path.join(os.path.dirname(file),
                                                           os.path.splitext(os.path.basename(file))[0] + '.' + ext)
        self.legacy = [
            {"sentences": [], "orig_text": [], "embeddings": [], "summary_onehot": [],
             "file_input": file,
             "file_summary": replace_extension(file, "summary.txt"),
             "file_pkl": replace_extension(file, str(limit or "all") + ".pkl"),
             "path": os.path.splitext(file)[0]
             }
            for file in list(glob.glob(self.pattern))]

    def trie(self):
        for d in self.documents():
            yield Trie(d.split(' '))

    from .explode import explode
    def tfidf(self, n=10):
        import numpy as np

        # docs = []
        for ps in self.generate(unit="paragraphs", detail=True):
            _ps = [(ps, idx) for ps, idx in ps[0]]
            ps = [x[0] for x in _ps]
            ps_idx = [x[1] for x in _ps]
            print(f"[document_model.tfidf]\n"
                  f"paragraph count: {len(ps)}\n"
                  f"sentence count per paragraph : {' '.join(ps).count('. ') / len(ps):.2f}\n")
            ## Determine TF-IDF parameters from bow
            # bow = dict(Trie(' '.join(ps).split(' ')).query("")))
            tf_inverse_cdf = lambda x: x
            idf_inverse_cdf = lambda y: -np.log(y)  # lambda y: np.log((1-y)/y)
            ## Count
            bow_size = int(np.sqrt(n * len(ps)))
            bow = {w: self.explode(w) for w in Trie(' '.join(ps).split(' ')).top_n_items(n=bow_size).keys()}
            ps = [self.explode(p) for p in ps]
            tfs = [{w: p.count(e) for w, e in bow.items()} for p in ps]
            idfs = {w: sum([(e in p) for p in ps]) for w, e in bow.items()}
            ## normalize & transform
            tfs = [dict(zip(tf.keys(), np.array(list(tf.values())) / max(1, sum(tf.values())))) for tf in tfs]
            idfs = dict(zip(idfs.keys(), np.array(list(idfs.values())) / max(1, sum(idfs.values()))))
            tfs = [{k: tf_inverse_cdf(v) for k, v in tf.items()} for tf in tfs]
            idfs = [{k: idf_inverse_cdf(v) for k, v in idfs.items()}] * len(ps)
            ## calculate tfidf
            tfidfs = [{k: tf[k] * idf[k] for k in tf.keys()} for tf, idf in zip(tfs, idfs)]

            ## Extract keywords
            tfidfs = [{k: int(100 * tfidf[k]) for k in sorted(tfidf.keys(), key=tfidf.get, reverse=True)[:n] if
                       tfidf[k] > 0.01} for tfidf in tfidfs]
            candidates = set(sum([list(tfidf.keys()) for tfidf in tfidfs], []))
            # Note: setdefault modifies dict
            tfidf_max = {x: max([tfidf.setdefault(x, 0) for tfidf in tfidfs]) for x in candidates}
            tfidf_count = {x: sum([(tfidf.setdefault(x, 0) > 0) for tfidf in tfidfs]) for x in candidates}
            tfidf_mean = {x: int(sum([tfidf.setdefault(x, 0) for tfidf in tfidfs]) / len(tfidfs)) for x in candidates}
            tfidf_max_mean = {x: int(100 * tfidf_mean[x] / (1 + tfidf_count[x])) for x in candidates}

            print(dict(sorted(tfidf_max.items(), key=lambda item: item[1], reverse=True)))
            print(dict(sorted(tfidf_count.items(), key=lambda item: item[1], reverse=True)))
            print(dict(sorted(tfidf_mean.items(), key=lambda item: item[1], reverse=True)))
            print(dict(sorted(tfidf_max_mean.items(), key=lambda item: item[1], reverse=True)))
            # candidates = {x:candidates.count(x) for x in set(candidates)}
            # print((sorted(candidates.keys(), key=candidates.get, reverse=True)))

            if True:
                import matplotlib.pyplot as plt
                import pandas as pd
                assert len(ps_idx) == len(tfidfs)

                # pd.DataFrame({
                #     "idx": [idx[3] for idx in ps_idx],
                #     "tfidf": [tfidf['시민'] for tfidf in tfidfs]}).query('tfidf != 0').plot.scatter(x="idx", y="tfidf", style='.')
                # plt.show()
                pd.DataFrame({
                    "idx": [idx[3] for idx in ps_idx],
                    "tfidf": [tfidf['디오'] for tfidf in tfidfs]}).query('tfidf != 0').plot.scatter(x="idx", y="tfidf", style='.')
                plt.xlim([0, max([idx[3] for idx in ps_idx])])
                plt.show()
                # pd.DataFrame({
                #     "idx": [idx[3] for idx in ps_idx],
                #     "tfidf": [tfidf['디오'] for tfidf in tfidfs]}).query('tfidf != 0').set_index('idx').plot.kde()
                # plt.show()
                pdf = zip([idx[3] for idx in ps_idx], [tfidf['디오'] for tfidf in tfidfs])
                samples = sum([[x]*p for x,p in pdf],[])
                pd.Series(samples).plot.kde()
                plt.xlim([0, max([idx[3] for idx in ps_idx])])
                plt.show()
            yield tfidfs

    def iter_text(self):
        """
        Global indexer
        :return: generator for (text, doc_id, paragraph_id, sentence_id, word_id)
        """
        raise NotImplementedError


def legacy_sentences_from_raw_text(path, limit=None, force=False):
    pkl_path = path + '.' + str(limit or 'all') + ".pkl"
    if glob.glob(pkl_path) and (not force):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
            tokens = data['tokens']
            text = data['text']
    else:
        print("POS tagging...")
        kkma = Kkma()
        from .preprocessing import legacy_preprocess
        with open(path, "r", encoding="utf-8") as reader:
            text = reader.readlines()
            if limit: text = text[0:limit]
            text = legacy_preprocess.__func__(''.join(text))
            text = kkma.sentences(text)
            tokens = [kkma.pos(x) for x in text]
            with open(pkl_path, "wb") as f:
                pickle.dump({'tokens': tokens, 'text': text}, f)
    return tokens, text
