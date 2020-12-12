## DL related
import os
import bert
import tensorflow as tf
from tensorflow import keras
import tflite_runtime.interpreter as tflite
import platform

## Tokenizer endcoder related
import pickle
import glob
from konlpy.tag import Kkma

tf.gfile = tf.io.gfile
from proprietary.korbert.tokenization_morp import load_vocab, convert_tokens_to_ids, \
    convert_ids_to_tokens  # proprietary/korbert/002_bert_morp_tensorflow/tokenization_morp.py
from collections import defaultdict
from pos_conversion_rule import POS_snu_tta

POS_convert = defaultdict(lambda: 'UN', POS_snu_tta)  # TODO : print warning

## Clutering related
from cluster_features import ClusterFeatures
import numpy as np


def preprocess(_text):
    def _whitespace(_text):
        t = _text.replace('\n', ' ')
        t = ' '.join(t.split())
        return t

    def _unknown_characters(_text):
        t = _text.replace('/', ' ')
        return t

    # _parse_page()
    t = _whitespace(_text)
    t = _unknown_characters(t)
    t = t.lower()
    return t


def sentences_from_raw_text(path, limit=None, force=False):
    pkl_path = path + '.' + str(limit or 'all') + ".pkl"
    if glob.glob(pkl_path) and (not force):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
            tokens = data['tokens']
            text = data['text']
    else:
        print("POS tagging...")
        kkma = Kkma()
        with open(path, "r", encoding="utf-8") as reader:
            text = reader.readlines()
            if limit: text = text[0:limit]
            text = preprocess(''.join(text))
            text = kkma.sentences(text)
            tokens = [kkma.pos(x) for x in text]
            with open(pkl_path, "wb") as f:
                pickle.dump({'tokens': tokens, 'text': text}, f)
    return tokens, text


def tokens_to_tensors(sentences, max_seq_len, vocab_file):
    sentences = [['[CLS]', *[x[0] + '/' + POS_convert[x[1]] + '_' for x in tokens], '[SEP]'] for tokens in sentences]
    sentences = [x[:min(len(x), max_seq_len) - 1] + ['[SEP]'] for x in sentences]
    sentences = [x + ['[PAD]'] * (max_seq_len - len(x)) for x in sentences]
    vocab = load_vocab(vocab_file)
    inv_vocab = {v: k for k, v in vocab.items()}
    vocab_wrap = defaultdict(lambda: vocab['/NA_'], vocab)  # FIXME : Fallback to '/NA_'
    sentence_vectors = [convert_tokens_to_ids(vocab_wrap, tokens) for tokens in sentences]  # TODO : Use FullTokenizer
    return sentence_vectors


class tpu_model(keras.Model):
    '''
    Model wrapper for TPU.
    '''
    batch_size = 1

    def build(self, *args, **kwargs):
        '''
        Make interpreter automatically after building model
        :param args:
        :param kwargs:
        :return:
        '''
        super().build(*args, **kwargs)
        device = []
        model_path = "proprietary/model.tflite"
        EDGETPU_SHARED_LIB = {
            'Linux': 'libedgetpu.so.1',
            'Darwin': 'libedgetpu.1.dylib',
            'Windows': 'edgetpu.dll'
        }[platform.system()]

        if glob.glob(model_path):
            # Load the TFLite model and allocate tensors.
            interpreter = tflite.Interpreter(model_path=model_path,
                                             experimental_delegates=[
                                                 tflite.load_delegate(EDGETPU_SHARED_LIB,
                                                                      {'device': device[0]} if device else {})])
        else:
            # Convert the model.
            model = self
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Accuracy loss

            tflite_model = converter.convert()
            # Save the model.
            with open(model_path, 'wb') as f:
                f.write(tflite_model)
            interpreter = tflite.Interpreter(model_content=tflite_model,
                                             experimental_delegates=[
                                                 tflite.load_delegate(EDGETPU_SHARED_LIB,
                                                                      {'device': device[0]} if device else {})])
        interpreter.allocate_tensors()
        # Get input and output tensors.
        input_details = interpreter.get_input_details()  # [{'name': 'input_1', 'index': 0, 'shape': array([ 1, 64], dtype=int32), 'shape_signature': array([-1, 64], dtype=int32), 'dtype': <class 'numpy.int32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]
        output_details = interpreter.get_output_details()  # [{'name': 'Identity', 'index': 1533, 'shape': array([  1, 768], dtype=int32), 'shape_signature': array([ -1, 768], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]
        # Resize input size
        interpreter.resize_tensor_input(input_details[0]['index'],
                                        [self.batch_size, input_details[0]['shape'][1]])
        interpreter.allocate_tensors()
        self.interpreter = interpreter
        self.input_details = input_details
        self.output_details = output_details

    def predict(self, input_data):
        '''
        Predict using TPU
        :param input_data:
        :return: output tensor
        '''
        interpreter = self.interpreter
        input_details = self.input_details
        output_details = self.output_details
        # # Test the model on random input data.
        # input_shape = input_details[0]['shape']
        # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # print(output_data.shape)
        return output_data


def load_bert(model_dir, model_file="model.ckpt", batch_size=32, max_seq_len=512, pooling="default"):
    model_ckpt = os.path.join(model_dir, model_file)
    bert_params = bert.params_from_pretrained_ckpt(model_dir)
    # max_seq_len = bert_params.max_position_embeddings

    l_bert = bert.BertModelLayer.from_params(bert_params)
    l_input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32')
    # l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32')

    # (1 segment) using the default token_type/segment id 0
    bert_output = l_bert(l_input_ids)  # output: [batch_size, max_seq_len, hidden_size]
    # Pooling layer for sentence vector
    if pooling == "default":  # First token ([CLS]) "This output is usually not a good summary of the semantic content ..."
        first_token_tensor = tf.squeeze(bert_output[:, 0:1, :], axis=1)
        output = tf.keras.layers.Dense(bert_params.hidden_size,
                                       activation=tf.tanh,
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                           stddev=bert_params.initializer_range))(first_token_tensor)
    if pooling == "average":
        output = tf.squeeze(
            tf.keras.layers.AveragePooling1D(pool_size=max_seq_len, data_format='channels_last')(bert_output), axis=1)
    elif pooling == "max":
        output = tf.squeeze(tf.keras.layers.MaxPool1D(pool_size=max_seq_len, data_format='channels_last')(bert_output),
                            axis=1)
    # else if pooling == "median" : # remove zeros and do something
    elif pooling == "none":
        output = bert_output
    if USE_TPU:
        model = tpu_model(inputs=l_input_ids, outputs=output)
        tpu_model.batch_size = batch_size
    else:
        model = keras.Model(inputs=l_input_ids, outputs=output)
    model.build(input_shape=(None, max_seq_len))

    # (2 segments) provide a custom token_type/segment id as a layer input
    # output = l_bert([l_input_ids, l_token_type_ids])  # [batch_size, max_seq_len, hidden_size]
    # model = keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=output)
    # model.build(input_shape=[(None, max_seq_len), (None, max_seq_len)])

    l_bert.apply_adapter_freeze()
    bert.load_stock_weights(l_bert, model_ckpt)
    return model


def embed_sentence(model, sentence_vectors, bert_params, batch_size, limit=None):
    embed_result = []
    if limit is None: limit = len(sentence_vectors)
    for offset in range(int(int(limit) / batch_size)):
        sentence_tensors = tf.convert_to_tensor(sentence_vectors[offset:offset + batch_size], dtype=tf.int32)
        import time
        start = time.perf_counter()
        result = model.predict(sentence_tensors)
        inference_time = time.perf_counter() - start
        print('%.1fms' % (inference_time * 1000))
        assert result.shape == (batch_size, bert_params.hidden_size)
        embed_result += result.tolist()
        print(f"Embedding : {len(embed_result)}/{limit}")
    # FIXME : truncating tails (batch size = 32), add them by padding
    return embed_result


raw_texts = [f"proprietary/text/{x}.txt" for x in ["공정하다는 착각", "생각에 관한 생각", "시지프 신화", "의료윤리", "행복의 기원"]]
vocab_file = 'proprietary/korbert/002_bert_morp_tensorflow/vocab.korean_morp.list'
model_dir = "proprietary/korbert/002_bert_morp_tensorflow/"
bert_params = bert.params_from_pretrained_ckpt(model_dir)
max_seq_len = 64  # bert_params.max_position_embeddings # 128 # 64
batch_size = 32
# pooling_method = 'average'
pooling_method = 'default'

summary_dir = "proprietary/summary/"
lines_to_read_limit = None  # 1000
summary_ratio = 0.1
summary_lines_override = None  # 100

USE_TPU = False #True # False
for raw_text_path in raw_texts:
    sentences, orig_text = sentences_from_raw_text(raw_text_path, limit=None, force=False)
    sentence_vectors = tokens_to_tensors(sentences, max_seq_len, vocab_file)
    model = load_bert(model_dir, model_file="model.ckpt", batch_size=batch_size, max_seq_len=len(sentence_vectors[0]),
                      pooling=pooling_method)
    embed_result = np.array(embed_sentence(model, sentence_vectors, bert_params, batch_size, limit=lines_to_read_limit))
    print("Clustering")
    summary_idx = ClusterFeatures(features=embed_result).cluster(ratio=summary_ratio,
                                                                 num_sentences=summary_lines_override)
    print("Clustering Done")
    summary = '\n'.join([orig_text[idx] for idx in summary_idx])
    summary_file_path = os.path.join(summary_dir, os.path.splitext(os.path.basename(raw_text_path))[0] + '.txt')
    with open(summary_file_path, "w") as f:
        f.write(summary)

# TODO : Check normalization of clustering input (because pooling method is average)
# FIXME : TPU optimization problem #@tf.function # https://www.tensorflow.org/guide/graph_optimization # print(tf.config.optimizer.get_experimental_options())
# TODO : Benchmark pooling methods
# TODO : Use FullTokenizer for preprocessing
# TODO : VALIDATION

debug = False
if debug:
    from sklearn.manifold import TSNE
    import pandas as pd
    import matplotlib.pyplot as plt

    perplexity = 50
    tsne_2d = TSNE(n_components=2, perplexity=perplexity)
    TCs_2d = pd.DataFrame(tsne_2d.fit_transform(pd.DataFrame(embed_result)))
    TCs_2d.columns = ["TC1_2d", "TC2_2d"]
    TCs_2d.plot.scatter("TC1_2d", "TC2_2d")
    plt.show()
