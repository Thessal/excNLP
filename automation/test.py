from encoder import *
from document_model import Document
from document_model.main import legacy_sentences_from_raw_text
from glob import glob

#
# Parameters
#
print("Setup")
paths = glob("proprietary/data/TEXT/Raw/EBOOK/*.txt")
path = [paths[0]]
read_line_limit = None  # 300
doc = Document(path[0], limit=read_line_limit)

#
# Document Model (Segmentation)
#
print("BOW")
for x in doc.trie():
    search = {w[0]: w[1] for w in x.query('승')}
    print(search)
    print(x.top_n_items(30))
    break

print("Document Model")
for res in doc.tfidf(n=15):
    for x in res:
        # print(list(x.keys()))
        pass
    break

# FIXME : multiple file handling (fix doc.documents in doc.tfidf rather than glob in test.py)
# TODO : coreference resolution
# TODO : save document model to file

# #
# # NER Train
# #
# print("Dataset load")
# from tool import dataset
# # dataset.generate_KMOU()
# # dataset.generate_CNU()
# # dataset.generate_NAVER()
# dfs1 = dataset.load_KMOU()
# dfs2 = dataset.load_CNU()
# dfs3 = dataset.load_NAVER()


# raw_texts = [f"proprietary/text/{x}.txt" for x in ["담론과 진실"]]
# silent = True
#
# max_seq_len = 64
# batch_size = 32
# pooling_method = 'default'  # 'average'
#
# summary_ratio = 0.03  # 0.1
# summary_lines_override = None  # 100
#
#
# #
# # Document Model (legacy)
# #
# print("Document Model (legacy)")
# raw_text_path = raw_texts[0]
#
# sentences, orig_text = legacy_sentences_from_raw_text(
#     raw_text_path, limit=read_line_limit, force=False)
# doc.legacy[0] = {"sentences": sentences, "orig_text": orig_text}
# print(doc.legacy[0]["orig_text"][0:100])
#
# #
# # Encoder
# #
# print("Encoder")
# e = Encoder(vocab_file='./proprietary/korbert/002_bert_morp_tensorflow/vocab.korean_morp.list',
#             model_dir="./proprietary/korbert/002_bert_morp_tensorflow/",
#             max_seq_len=64,
#             batch_size=32,
#             pooling_method='default',  # 'average'
#             silent=True,
#             )
# print("Encoder - Encode")
# e.encode(doc.legacy[0])
# print(doc.legacy[0]["embeddings"][0][0:5])
#
# print("Encoder - Cluster")
# c = ClusterFeaturesSummarizer(summary_ratio=summary_ratio, summary_lines_override=summary_lines_override)
# c.summarize(doc.legacy[0], save_to_file=False)
# print(doc.legacy[0]["summary"][0:30])
#
# # TODO : Check normalization of clustering input (because pooling method is average)
# # FIXME : TPU optimization problem #@tf.function # https://www.tensorflow.org/guide/graph_optimization # print(tf.config.optimizer.get_experimental_options())
# # TODO : Benchmark pooling methods
# # TODO : Use FullTokenizer for preprocessing
# # TODO : VALIDATION
# # Done loading 196 BERT weights from: ./proprietary/korbert/002_bert_morp_tensorflow/model.ckpt into <bert.model.BertModelLayer object at 0x147e81fa0> (prefix:bert_model_layer). Count of weights not found in the checkpoint was: [0]. Count of weights with mismatched shape: [0]
# # Unused weights from checkpoint:
# # 	{...}/adam_m
# # 	{...}/adam_v
#
# #
# # Debug
# #
# print("Encoder Debug")
# debug = False
# if debug:
#     from sklearn.manifold import TSNE
#     import pandas as pd
#     import matplotlib.pyplot as plt
#
#     perplexity = 50
#     tsne_2d = TSNE(n_components=2, perplexity=perplexity)
#     TCs_2d = pd.DataFrame(tsne_2d.fit_transform(pd.DataFrame(embed_result)))
#     TCs_2d.columns = ["TC1_2d", "TC2_2d"]
#     TCs_2d.plot.scatter("TC1_2d", "TC2_2d")
#     plt.show()
