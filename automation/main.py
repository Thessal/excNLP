## DL related
import os
import bert
import tensorflow as tf
from tensorflow import keras

## Tokenizer endoder related
import pickle
import glob
from konlpy.tag import Kkma

tf.gfile = tf.io.gfile
from proprietary.korbert.tokenization_morp import load_vocab, convert_tokens_to_ids, convert_ids_to_tokens # proprietary/korbert/002_bert_morp_tensorflow/tokenization_morp.py
from collections import defaultdict
from pos_conversion_rule import POS_snu_tta

POS_convert = POS_snu_tta


def preprocess(_text):
    def _whitespace(_text):
        t = _text.replace('\n', ' ')
        t = ' '.join(_text.split())
        return t

    # _parse_page()
    t = _whitespace(_text)
    # _unknown_characters()
    t = t.lower()
    return t


def sentences_from_raw_text(path, limit=None, force=False):
    pkl_path = path + '.' + str(limit or 'all') + ".pkl"
    if glob.glob(pkl_path) and (not force):
        with open(pkl_path, "rb") as f:
            tokens = pickle.load(f)
    else:
        kkma = Kkma()
        with open(path, "r", encoding="utf-8") as reader:
            text = reader.readlines()
            if limit: text = text[0:limit]
            text = preprocess(''.join(text))
            text = kkma.sentences(text)
            tokens = [kkma.pos(x) for x in text]
            with open(pkl_path, "wb") as f:
                pickle.dump(tokens, f)
    return tokens


def load_bert(max_seq_len=512):
    model_dir = "proprietary/korbert/002_bert_morp_tensorflow/"
    model_ckpt = os.path.join(model_dir, "model.ckpt")

    bert_params = bert.params_from_pretrained_ckpt(model_dir)
    l_bert = bert.BertModelLayer.from_params(bert_params)

    l_input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32')
    #l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32')

    # (1 segment) using the default token_type/segment id 0
    output = l_bert(l_input_ids)                              # output: [batch_size, max_seq_len, hidden_size]
    model = keras.Model(inputs=l_input_ids, outputs=output)
    model.build(input_shape=(None, max_seq_len))

    # (2 segments) provide a custom token_type/segment id as a layer input
    # output = l_bert([l_input_ids, l_token_type_ids])  # [batch_size, max_seq_len, hidden_size]
    # model = keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=output)
    # model.build(input_shape=[(None, max_seq_len), (None, max_seq_len)])

    l_bert.apply_adapter_freeze()
    bert.load_stock_weights(l_bert, model_ckpt)
    return model


raw_texts = [f"proprietary/text/{x}.txt" for x in ["공정하다는 착각", "생각에 관한 생각", "시지프 신화", "의료윤리", "행복의 기원"]]
vocab_file = 'proprietary/korbert/002_bert_morp_tensorflow/vocab.korean_morp.list'
max_seq_len = 64
batch_size = 32

sentences = sentences_from_raw_text(raw_texts[3], limit=None, force=False)
sentences = [['[CLS]', *[x[0] + '/' + POS_convert[x[1]] + '_' for x in tokens], '[SEP]'] for tokens in sentences]
sentences = [x[:min(len(x), max_seq_len)-1]+['[SEP]'] for x in sentences]
sentences = [x+['[PAD]']*(max_seq_len-len(x)) for x in sentences]
# print(sentences[0])
vocab = load_vocab(vocab_file)
inv_vocab = {v: k for k, v in vocab.items()}
vocab_wrap = defaultdict(lambda: 13243, vocab)  # FIXME : Fallback to '/NA_'
sentence_vectors = [convert_tokens_to_ids(vocab_wrap, tokens) for tokens in sentences] # TODO : Use FullTokenizer
model = load_bert(max_seq_len)
embed_result = []
#for offset in range(int(len(sentence_vectors)/batch_size)):
for offset in range(1):
    sentence_tensors = tf.convert_to_tensor(sentence_vectors[offset:offset+batch_size], dtype=tf.int32)
    embed_result += model.predict(sentence_tensors).tolist()

sentence_idx = 0
word_idx = 1
print([int(x*100) for x in embed_result[sentence_idx][word_idx]])