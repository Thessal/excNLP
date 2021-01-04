from constant import *
import os
import json
from tokenizer import *
from encoder import BertModel
from encoder import Encoder
from encoder import ClusterFeaturesSummarizer
from document_model import Document
from document_model.main import legacy_sentences_from_raw_text

rebuild = False

#
# Tokenizer
#
tokenizer_trie = Tokenizer(TOKENIZER_DIR, text_files=RAW_TEXT_FILES) # need to make abstract train method, need to remove text_files option
tokenizer_sentencepiece = TokenizerSpm(
    TOKENIZER_DIR,
    train_args={
        "files": RAW_TEXT_FILES_SHUFFLE,
        "character_coverage": 0.9995,
        "vocab_size": 200000,
    }
)
tokenizer = tokenizer_sentencepiece

if rebuild :
    tokenizer.train(delete_previous_file=True, chunksize=1000000) #chunksize need to be large enough than vocab_size
tokenizer.load(enable_tf=False)
tokenizer.tokenize(["싸움하는사람은즉싸움하지아니하던사람이고또싸움하는사람은싸움하지아니하는사람이었기도하니까싸움하는사람이싸움하는구경을하고싶거든싸움하지아니하던사람이싸움하는것을구경하든지싸움하지아니하는사람이싸움하는구경을하든지싸움하지아니하던사람이나싸움하지아니하는사람이싸움하지아니하는것을구경하든지하였으면그만이다"],debug=True)
tokenizer.tokenize(["구원적거의지의일지·일지에피는현화·특이한사월의화초·삼십륜·삼십륜에전후되는양측의명경·맹아와같이희희하는지평을향하여금시금시낙탁하는 만월·청간의기가운데 만신창이의만월이의형당하여혼륜하는·적거의지를관류하는일봉가신·나는근근히차대하였더라·몽몽한월아·정밀을개엄하는대기권의요원·거대한곤비가운데의일년사월의공동·반산전도하는성좌와 성좌의천렬된사호동을포도하는거대한풍설·강매·혈홍으로염색된암염의분쇄·나의뇌를피뢰침삼아 침하반과되는광채임리한망해·나는탑배하는독사와같이 지평에식수되어다시는기동할수없었더라·천량이올때까지"],debug=True)
tokenizer.tokenize(["男子와 女子의","아랫도리가 젖어 있다.","밤에 보는 오갈피나무.","오갈피나무의 아랫도리가 젖어 있다.","맨발로 바다를 밟고 간 사람은","새가 되었다고 한다.","발바닥만 젖어 있었다고 한다."],debug=True)
tokenizer.tokenize(["아이~씻팔!!!", "초능력 맛 좀 볼래?", "좆같은도마뱀새끼", "경찰에 신고하거나 하면 희동이 호로자식 되는거야 알지? 처신 잘하라고", "어이 둘리.", "도우너 어서오고.", "아침부터 왜 이렇게 죽상이야.", "고길동이 꼴받게 하잖아 씨팔 젓밥새끼가", "ㅋㅋ", "떨 한 대 할래?", "좋지. 한 대 말아줘", "응? 콜록 콜록 아이고 이게 무슨 냄새야", "둘리!!! 집 안에서는 담배피지 말라고 했잖아!! 희동이도 있는데!!"],debug=True)
tokenizer.tokenize(["세상에는 두세 개의 죄악만이 존재하고 있고, 나머지는 부수적인 것들이오.", "그로부터는 더 이상 아무런 감흥도 받을 수 없소.", "빌어먹을, 내가 얼마나 여러 번, 태양을 공격할 수 있기를, 태양으로부터 우주를 빼앗기를, 그래서 그 태양을 사용하여 이 지구를 불태워버리기를 염원한지 아시오?", "그야말로 완벽한 죄악이지요.", "우리가 지금 몰두하고 있는 것처럼 작은 일탈행위가 아니라 일 년 내에 족히 12명의 피조물을 흙덩어리로 변화시켜 버리기에 충분한 진짜 죄악이란 말이오."],debug=True)

raise ValueError
#
# Embedding
#
embedder = BertModel()
if rebuild:
    # Prepare data.
    files_in = RAW_TEXT_FILES
    files_out = [x.replace("/TEXT/Raw/", "/EMBED/BERT/") for x in RAW_TEXT_FILES]  # need to use .tfrecord
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


#
# WIP : BERT embedding, clustering
# Need to rewrite code and remove korBERT
# Working..kind of
import shutil
shutil.copy("proprietary/data/EMBED/BERT_CHECKPOINT/model.ckpt-100000.data-00000-of-00001",
            "proprietary/data/EMBED/BERT_MODEL/model.ckpt.data-00000-of-00001")
shutil.copy("proprietary/data/EMBED/BERT_CHECKPOINT/model.ckpt-100000.index",
            "proprietary/data/EMBED/BERT_MODEL/model.ckpt.index")
shutil.copy("proprietary/data/EMBED/BERT_CHECKPOINT/model.ckpt-100000.meta",
            "proprietary/data/EMBED/BERT_MODEL/model.ckpt.meta")
e = Encoder(vocab_file='./proprietary/data/EMBED/BERT_MODEL/bert.vocab',
            model_dir="./proprietary/data/EMBED/BERT_MODEL",
            max_seq_len=128,
            batch_size=32,
            pooling_method='default',  # 'average'
            silent=True,
            )

# print("Encoder - Cluster")
sentences, orig_text = legacy_sentences_from_raw_text(
    "proprietary/data/TEXT/Raw/EBOOK/담론과 진실.txt", limit=300, force=True)
doc = {"sentences": sentences, "orig_text": orig_text}
e.encode(doc)
c = ClusterFeaturesSummarizer(summary_ratio=0.1, summary_lines_override=None)
c.summarize(doc, save_to_file=False)
print(doc["summary"])

# #
# # Document Model (Segmentation)
# #
# paths = glob("proprietary/data/TEXT/Raw/EBOOK/*.txt")
# path = [paths[0]]
# read_line_limit = None  # 300
# doc = Document(path[0], limit=read_line_limit)
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

