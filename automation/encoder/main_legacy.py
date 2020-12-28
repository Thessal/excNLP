## DL related
import os
import bert
import tensorflow as tf
from tensorflow import keras
#import tflite_runtime.interpreter as tflite
import platform

tf.gfile = tf.io.gfile
from proprietary.korbert.tokenization_morp import load_vocab, \
    convert_tokens_to_ids  # proprietary/korbert/002_bert_morp_tensorflow/tokenization_morp.py
from collections import defaultdict
from tokenizer.pos_conversion_rule import POS_snu_tta
import glob

POS_convert = defaultdict(lambda: 'UN', POS_snu_tta)  # TODO : print warning

## Clutering related
import numpy as np
class Encoder:
    def __init__(self,
                 vocab_file='./proprietary/korbert/002_bert_morp_tensorflow/vocab.korean_morp.list',
                 model_dir="./proprietary/korbert/002_bert_morp_tensorflow/",
                 model_file="model.ckpt",
                 max_seq_len=64,
                 batch_size=32,
                 pooling_method='default',
                 USE_TPU=False,  # True, # False
                 silent=False
                 ):
        """
        Convert text to vector using BERT model
        :param vocab_file: POS tagged dictionary for model
        :param model_dir: pre-trained model
        :param max_seq_len: upper limit of token count for any sentence
        :param batch_size: tensorflow run batch size (Usually 32 for CPU, 1 for TPU)
        :param lines_to_read_limit: (legacy, remove me) lines to read from file
        :param USE_TPU: Utilize coral TPU if True
        """
        self.vocab_file = vocab_file
        self.model_dir = model_dir
        self.model_file = model_file
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.USE_TPU = USE_TPU
        self.bert_params = bert.params_from_pretrained_ckpt(model_dir)
        self.max_seq_len = min(self.bert_params.max_position_embeddings, max_seq_len)
        self.pooling = pooling_method  # 'average'
        if silent :
            bert.loader._verbose = 0
            bert.loader.trace = lambda *a, **k: None
        self.model = self.load_bert()

    def encode(self, legacy_document):
        sentence_vectors = self.tokens_to_tensors(legacy_document["sentences"])
        embed_result = np.array(
            self.embed_sentence(sentence_vectors))
        legacy_document["embeddings"] = embed_result

    def tokens_to_tensors(self, sentences):
        sentences = [['[CLS]', *[x[0] + '/' + POS_convert[x[1]] + '_' for x in tokens], '[SEP]'] for tokens in
                     sentences]
        sentences = [x[:min(len(x), self.max_seq_len) - 1] + ['[SEP]'] for x in sentences]
        sentences = [x + ['[PAD]'] * (self.max_seq_len - len(x)) for x in sentences]
        vocab = load_vocab(self.vocab_file)
        inv_vocab = {v: k for k, v in vocab.items()}
        vocab_wrap = defaultdict(lambda: vocab['/NA_'], vocab)  # FIXME : Fallback to '/NA_'
        sentence_vectors = [convert_tokens_to_ids(vocab_wrap, tokens) for tokens in
                            sentences]  # TODO : Use FullTokenizer
        return sentence_vectors

    def load_bert(self):
        # https://gist.github.com/batzner/7c24802dd9c5e15870b4b56e22135c96
        model_ckpt = os.path.join(self.model_dir, self.model_file)
        bert_params = bert.params_from_pretrained_ckpt(self.model_dir)
        # max_seq_len = bert_params.max_position_embeddings

        l_bert = bert.BertModelLayer.from_params(bert_params)
        l_input_ids = keras.layers.Input(shape=(self.max_seq_len,), dtype='int32')
        # l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32')

        # (1 segment) using the default token_type/segment id 0
        bert_output = l_bert(l_input_ids)  # output: [batch_size, max_seq_len, hidden_size]
        # Pooling layer for sentence vector
        if self.pooling == "default":  # First token ([CLS]) "This output is usually not a good summary of the semantic content ..."
            first_token_tensor = tf.squeeze(bert_output[:, 0:1, :], axis=1)
            output = tf.keras.layers.Dense(bert_params.hidden_size,
                                           activation=tf.tanh,
                                           kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                               stddev=bert_params.initializer_range))(first_token_tensor)
        if self.pooling == "average":
            output = tf.squeeze(
                tf.keras.layers.AveragePooling1D(pool_size=self.max_seq_len, data_format='channels_last')(bert_output),
                axis=1)
        elif self.pooling == "max":
            output = tf.squeeze(
                tf.keras.layers.MaxPool1D(pool_size=self.max_seq_len, data_format='channels_last')(bert_output),
                axis=1)
        # else if pooling == "median" : # remove zeros and do something
        elif self.pooling == "none":
            output = bert_output
        if self.USE_TPU:
            model = tpu_model(inputs=l_input_ids, outputs=output)
            tpu_model.batch_size = self.batch_size
        else:
            model = keras.Model(inputs=l_input_ids, outputs=output)
        model.build(input_shape=(None, self.max_seq_len))

        # (2 segments) provide a custom token_type/segment id as a layer input
        # output = l_bert([l_input_ids, l_token_type_ids])  # [batch_size, max_seq_len, hidden_size]
        # model = keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=output)
        # model.build(input_shape=[(None, max_seq_len), (None, max_seq_len)])

        l_bert.apply_adapter_freeze()
        bert.load_stock_weights(l_bert, model_ckpt)

        # load the fine tuned classifier model weights
        # fine_path = os.path.join(self.model_dir, self.model_file)
        # finetuned_weights = {w.name: w for w in model.trainable_weights}
        # print(finetuned_weights)
        # checkpoint = tf.train.Checkpoint(**finetuned_weights)
        # load_status = checkpoint.restore(fine_path)
        # load_status.assert_consumed().run_restore_ops()

        return model

    def embed_sentence(self, sentence_vectors):
        embed_result = []
        limit = len(sentence_vectors)
        for offset in range(int(int(limit) / self.batch_size)):
            sentence_tensors = tf.convert_to_tensor(sentence_vectors[offset:offset + self.batch_size], dtype=tf.int32)
            import time
            start = time.perf_counter()
            result = self.model.predict(sentence_tensors)
            inference_time = time.perf_counter() - start
            assert result.shape == (self.batch_size, self.bert_params.hidden_size)
            embed_result += result.tolist()
            bert.loader.trace(f"Embedding : {len(embed_result)}/{limit} ({(inference_time * 1000):.1f}ms)")
        # FIXME : truncating tails (batch size = 32), add them by padding
        return embed_result


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
        model_path = "../proprietary/model.tflite"
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
