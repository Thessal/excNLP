# from util.explode import explode
#
#
# def tfidf(self, n=10):
#     import numpy as np
#     for ps in self.generate(unit="paragraphs", detail=True):
#         word_offset = [p["begin"][3] for p in ps["element"]] + [ps["end"][3]]
#         word_count = [x - y for x, y in zip(word_offset[1:], word_offset[:-1])]
#         sentence_offset = [p["begin"][2] for p in ps["element"]] + [ps["end"][2]]
#         sentence_count = [x - y for x, y in zip(sentence_offset[1:], sentence_offset[:-1])]
#
#         ps = [p["text"] for p in ps["element"]]
#         print(f"[document_model.tfidf]\n"
#               f"paragraph count: {len(ps)}\n"
#               f"sentence count per paragraph : {' '.join(ps).count('. ') / len(ps):.2f}\n")
#
#         ## Determine TF-IDF parameters from bow
#         # bow = dict(Trie(' '.join(ps).split(' ')).query("")))
#         tf_inverse_cdf = lambda y: 0 if y > 0.1 else min(y, 0.05)  # FIXME : Magic numbers
#         idf_inverse_cdf = lambda y: -np.log(y)  # lambda y: np.log((1-y)/y)
#         ## Count
#         bow_size = int(np.sqrt(n * len(ps)))
#         subword = False  # We need a better candidate heuristic to enable this.
#         if subword:
#             bow = {w: self.explode(w) for w in Trie(' '.join(ps).split(' ')).top_n_items(n=bow_size).keys()}
#             # ps = [' '.join([self.explode(w) for w in p.split(' ')]) for p in ps]
#             ps = [self.explode(p) for p in ps]
#             print(bow)
#         else:
#             bow = {w: w for w in Trie(' '.join(ps).split(' '), auto_explode=False).top_n_items(n=bow_size).keys()}
#         tfs = [{w: p.count(e) for w, e in bow.items()} for p in ps]
#         idfs = {w: sum([(e in p) for p in ps]) for w, e in bow.items()}
#         ## normalize & transform
#         assert len(tfs) == len(word_count)
#         tfs = [dict(zip(tf.keys(), np.array(list(tf.values())) / max(1, wc))) for tf, wc in zip(tfs, word_count)]
#         idfs = dict(zip(idfs.keys(), np.array(list(idfs.values())) / max(1, sum(idfs.values()))))
#         tfs = [{k: tf_inverse_cdf(v) for k, v in tf.items()} for tf in tfs]
#         idfs = [{k: idf_inverse_cdf(v) for k, v in idfs.items()}] * len(ps)
#         ## calculate tfidf
#         tfidfs = [{k: tf[k] * idf[k] for k in tf.keys()} for tf, idf in zip(tfs, idfs)]
#         ## Weighted tfidfs
#         assert len(tfidfs) == len(sentence_count)
#         weight = [w ** 0.5 for w in sentence_count]
#         tfidfs = [dict(zip(x.keys(), [v * w for v in x.values()])) for w, x in zip(weight, tfidfs)]
#         tfidfs = [{k: int(100 * tfidf[k]) for k in sorted(tfidf.keys(), key=tfidf.get, reverse=True)[:n] if
#                    tfidf[k] > 0.01} for tfidf in tfidfs]  # FIXME : Magic numbers
#         ## Extract keywords
#         candidates = set(sum([list(tfidf.keys()) for tfidf in tfidfs], []))
#         [[tfidf.setdefault(x, 0) for x in candidates] for tfidf in tfidfs]  # sets default
#         candidates_tfs = {x: sum([tf[x] for tf in tfs]) for x in candidates}
#         for k in candidates:  # FIXME : Heuristic
#             subwords = [(n + 1, k[:-n - 1]) for n in range(len(k) - 1)]
#             for level, subword in subwords:
#                 if subword in candidates_tfs.keys():
#                     if candidates_tfs[subword] > candidates_tfs[k] * (3 ** level):  # 도시는, 도시를, 도시가, 도시., 도시다, ...
#                         candidates = [x for x in candidates if x != k]
#                     if candidates_tfs[subword] < candidates_tfs[k] * (1 + 0.1 * level):  # 디오, 디오게, 디오게네, 디오게네스
#                         candidates = [x for x in candidates if x != subword]
#         n_topic_segmentation = 10
#         n_topic_visualization = 5
#         if True:
#             from scipy.stats import gaussian_kde
#             import matplotlib.pyplot as plt
#             import pandas as pd
#             ## KDE
#             keywords = candidates
#             paragraph_position = [idx for idx in word_offset[:-1]]
#             x_grid = np.arange(0, word_offset[-1],
#                                int(word_offset[-1] / 100))  # np.linspace(0, word_offset[-1], 100)
#             pdfs = [zip(paragraph_position, [tfidf[keyword] for tfidf in tfidfs]) for keyword in keywords]
#             samples = [sum([[x] * p for x, p in pdf], []) for pdf in pdfs]  # FIXME : sampling from pdf... wasteful
#             kdes = [gaussian_kde(sample, bw_method='scott')(x_grid) for sample in samples]
#             df = pd.DataFrame(dict(zip(keywords, kdes)), index=x_grid)
#             ## Keyword selection from maximum probability
#             keywords = df.max(axis=0).nlargest(n_topic_segmentation).index
#             df[keywords[:min(n_topic_segmentation, n_topic_visualization)]].plot()
#             plt.legend(prop={"family": "AppleGothic"})
#             plt.show()
#
#             ## Segmentation
#             major_topics = df[keywords].idxmax(axis=1)
#             boundaries = major_topics.loc[major_topics.shift(1) != major_topics]
#             boundaries = list(boundaries.index) + [df.index[-1]]
#             boundaries = [boundaries[i] for i in range(len(boundaries) - 1)
#                           if (boundaries[i + 1] - boundaries[i] > 0.05 * boundaries[-1])] + [
#                              boundaries[-1]]  # FIXME : magic number
#             segments = [(x, y) for x, y in zip(boundaries[:-1], boundaries[1:])]
#             topics = [df.loc[x:y].mean(axis=0).nlargest(n_topic_visualization) for x, y in segments]
#             # print([list(topic.index) for topic in topics])
#             ## Plot
#             flip = True
#             x = np.array(boundaries)
#             y = np.array([np.array([t.values[n] for t in topics]) for n in range(n_topic_visualization)])
#             if flip: y = np.flipud(y)
#             b = np.zeros(y.shape)
#             b[1:] = np.cumsum(y[:-1], axis=0)
#             c = [[(0, 0, 1, x), (0, 1, 0, x), (0, 1, 1, x), (1, 0, 0, x), (1, 0, 1, x), (1, 1, 0, x)] for x in
#                  np.linspace(1, 0.1, n_topic_visualization)]
#             if flip: c = np.flipud(c)
#             t = np.array([np.array([t.index[n] for t in topics]) for n in range(n_topic_visualization)])
#             if flip: t = np.flipud(t)
#             p = [None] * n_topic_visualization
#             for n in range(n_topic_visualization):
#                 p[n] = plt.bar(x[:-1], y[n], width=x[1:] - x[:-1], align='edge', color=c[n], bottom=b[n],
#                                edgecolor="black")
#
#             for i in range(len(segments)):
#                 for j in range(n_topic_visualization):
#                     plt.text((x[i] + x[i + 1]) * 0.5, (0.5 * y + b)[j][i], t[j][i], ha='center', va='center',
#                              family="AppleGothic",
#                              bbox=dict(facecolor=(1, 1, 1, 0.5), edgecolor='None', pad=0.0))
#             # consider hash text into RB and segment index into G.
#             # text wrap
#             plt.show()
#         yield tfidfs