from constant import *
from tokenizer import Tokenizer
from tokenizer import TokenizerSpm
from encoder import BertModel
from document_model import Document
from document_model.main import legacy_sentences_from_raw_text

# import datetime
# import argparse

rebuild = False

# tokenizer = TokenizerSpm(
#     TOKENIZER_DIR,
#     train_args={
#         "files": RAW_TEXT_FILES_SHUFFLE,
#         "character_coverage": 0.9995,
#         "vocab_size": 200000,
#     }
# )
# if rebuild:
#     tokenizer.train(delete_previous_file=True, chunksize=1000000) #chunksize need to be large enough than vocab_size
# tokenizer.load(enable_tf=False)
# tokenizer.tokenize(["나는 자랑스러운 태극기 앞에 자유롭고 정의로운 대한민국의 무궁한 영광을 위하여 충성을 다할 것을 굳게 다짐합니다"])
# tokenizer.tokenize(["싸움하는사람은즉싸움하지아니하던사람이고또싸움하는사람은싸움하지아니하는사람이었기도하니까싸움하는사람이싸움하는구경을하고싶거든싸움하지아니하던사람이싸움하는것을구경하든지싸움하지아니하는사람이싸움하는구경을하든지싸움하지아니하던사람이나싸움하지아니하는사람이싸움하지아니하는것을구경하든지하였으면그만이다"])
# tokenizer.tokenize(["구원적거의지의일지·일지에피는현화·특이한사월의화초·삼십륜·삼십륜에전후되는양측의명경·맹아와같이희희하는지평을향하여금시금시낙탁하는 만월·청간의기가운데 만신창이의만월이의형당하여혼륜하는·적거의지를관류하는일봉가신·나는근근히차대하였더라·몽몽한월아·정밀을개엄하는대기권의요원·거대한곤비가운데의일년사월의공동·반산전도하는성좌와 성좌의천렬된사호동을포도하는거대한풍설·강매·혈홍으로염색된암염의분쇄·나의뇌를피뢰침삼아 침하반과되는광채임리한망해·나는탑배하는독사와같이 지평에식수되어다시는기동할수없었더라·천량이올때까지"])
# tokenizer.tokenize(["男子와 女子의","아랫도리가 젖어 있다.","밤에 보는 오갈피나무.","오갈피나무의 아랫도리가 젖어 있다.","맨발로 바다를 밟고 간 사람은","새가 되었다고 한다.","발바닥만 젖어 있었다고 한다."])
# tokenizer.tokenize(["아이~씻팔!!!", "초능력 맛 좀 볼래?", "좆같은도마뱀새끼", "경찰에 신고하거나 하면 희동이 호로자식 되는거야 알지? 처신 잘하라고", "어이 둘리.", "도우너 어서오고.", "아침부터 왜 이렇게 죽상이야.", "고길동이 꼴받게 하잖아 씨팔 젓밥새끼가", "ㅋㅋ", "떨 한 대 할래?", "좋지. 한 대 말아줘", "응? 콜록 콜록 아이고 이게 무슨 냄새야", "둘리!!! 집 안에서는 담배피지 말라고 했잖아!! 희동이도 있는데!!"])

import os
import json
embedder = BertModel()
rebuild=True
if rebuild:
    # Prepare data. 355 files are ignored
    files_in = RAW_TEXT_FILES
    files_out = [x.replace("/TEXT/Raw/", "/EMBED/BERT/") for x in RAW_TEXT_FILES]
    unprocessed = [i for i,file in enumerate(files_out) if not os.path.isfile(file)]
    files_in = [files_in[i] for i in unprocessed]
    files_out = [files_out[i] for i in unprocessed]
    print(f"Processing : {len(files_out)} / {len(RAW_TEXT_FILES)}")
    vocab_file = 'proprietary/data/POS/sentencepiece.vocab'
    embedder.create_pretraining_data(files_in, files_out, vocab_file, converted_vocab_file=BERT_VOCAB_FILE, pool_size=7)
    # Train
    with open(BERT_VOCAB_FILE,"r") as f:
        BERT_BASE_CONFIG['vocab_size'] = len(f.readlines())
    bert_config_file = f"{BERT_MODEL_DIR}/bert_config.json"
    with open(bert_config_file, "w") as fo:
        json.dump(BERT_BASE_CONFIG, fo, indent=2)
    input_files = [x.replace("/TEXT/Raw/", "/EMBED/BERT/") for x in RAW_TEXT_FILES]
    input_files = [file for file in input_files if os.path.isfile(file)]
    embedder.run_pretraining(bert_config_file,input_files,BERT_CHECKPOINT_DIR)

# #
# # Parameters
# #
# print("Setup")
# paths = glob("proprietary/data/TEXT/Raw/EBOOK/*.txt")
# path = [paths[0]]
# read_line_limit = None  # 300
# doc = Document(path[0], limit=read_line_limit)
#
# #
# # Document Model (Segmentation)
# #
# print("BOW")
# for x in doc.trie():
#     search = {w[0]: w[1] for w in x.query('승')}
#     print(search)
#     print(x.top_n_items(30))
#     break
#
# print("Document Model")
# for res in doc.tfidf(n=15):
#     for x in res:
#         # print(list(x.keys()))
#         pass
#     break
#
# # FIXME : multiple file handling (fix doc.documents in doc.tfidf rather than glob in test.py)
# # TODO : coreference resolution
# # TODO : save document model to file

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
