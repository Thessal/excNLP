from tokenizer import *
from constant import *
from document_model.file_io import TextIO
import io

# for fp in files:
#     lines = [x.strip() for x in fp.readlines()]
#     lines = TextIO.heuristic_formatting(lines, debug=False)

# a = io.BytesIO()
# a.write("hello".encode())
# txt = a.getvalue()
# txt = txt.decode("utf-8")
# print(txt)

class BertModel:
    def __init__(self):
        self.shape=(1,128)
        self.tokenizer = TokenizerSpm(
            TOKENIZER_DIR
        )
        self.tokenizer.load(enable_tf=False)
        return

    def bert_preprocess_model(self,text):
        #[PAD] 200000
        #[UNK] 0
        #[CLS] 200001
        #[SEP] 200002
        #[MASK] 200003
        #< S > 1
        #< T > 2
        N = self.shape[1]
        output = [y for x in self.tokenizer.tokenize(text) for y in x]
        output = output[:min(len(output), N)]
        return {
            'input_mask': (output + [200000] * (N-len(output))),
            'input_word_ids': [1]*len(output)+[0]*(N-len(output)),
            'input_type_ids': [0]*len(output),
        }

    def create_pretraining_data(self,files_in,files_out,vocab_file):
        assert(len(files_in)==len(files_out))
        textio = TextIO(None)

        # Monkey patch
        import tensorflow as tf
        tf.flags = tf.compat.v1.flags
        tf.python_io = tf.compat.v1.python_io
        tf.logging = tf.compat.v1.logging
        tf.gfile = tf.compat.v1.gfile
        import os
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + "/encoder/bert")
        from .bert import create_pretraining_data as pre

        pre.FLAGS.vocab_file=vocab_file
        pre.FLAGS.do_lower_case = False
        pre.FLAGS.do_whole_word_mask = False
        pre.FLAGS.max_seq_length = 128
        pre.FLAGS.max_predictions_per_seq = 20
        pre.FLAGS.random_seed = 12345
        pre.FLAGS.dupe_factor = 10
        pre.FLAGS.masked_lm_prob = 0.15
        pre.FLAGS.short_seq_prob = 0  # ", 0.1,

        ## TODO : pre.main.tokenizer.tokenize = self.tokenizer.tokenize

        for file_in, file_out in zip(files_in,files_out):
            # Convert to BERT format
            with open(file_in, "r") as fp:
                lines = [x.strip() for x in fp.readlines()]
                output = '\n'.join(textio.heuristic_formatting(lines))
                with open("/dev/shm/temp.txt", "w", encoding="utf-8") as tmpf: #¯\_(ツ)_/¯
                    tmpf.write(output)
            # Create pretraining data
            pre.FLAGS.input_file = "/dev/shm/temp.txt"
            pre.FLAGS.output_file = file_out
            pre.main(None)




# from bert import BertModelLayer
#
# l_bert = BertModelLayer(BertModelLayer.Params(
#   vocab_size               = 16000,        # embedding params
#   use_token_type           = True,
#   use_position_embeddings  = True,
#   token_type_vocab_size    = 2,
#
#   num_layers               = 12,           # transformer encoder params
#   hidden_size              = 768,
#   hidden_dropout           = 0.1,
#   intermediate_size        = 4*768,
#   intermediate_activation  = "gelu",
#
#   adapter_size             = None,         # see arXiv:1902.00751
#
#   name                     = "bert"        # any other Keras layer params
# ))








