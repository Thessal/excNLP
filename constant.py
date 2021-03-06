from glob import glob
import random
RAW_TEXT_FILES = glob("proprietary/data/TEXT/Raw/**/*.txt", recursive=True)
TOKENIZER_DIR = "proprietary/data/POS/"
RAW_TEXT_FILES_SHUFFLE = (RAW_TEXT_FILES*6).copy()
random.shuffle(RAW_TEXT_FILES_SHUFFLE)
BERT_VOCAB_FILE = "proprietary/data/EMBED/BERT_MODEL/bert.vocab" #"/dev/shm/temp.vocab"
BERT_MODEL_DIR = "proprietary/data/EMBED/BERT_MODEL"
BERT_CHECKPOINT_DIR = "proprietary/data/EMBED/BERT_CHECKPOINT"
I_AM_POOR = True
BERT_BASE_CONFIG = { # You need to delete checkpoint after changing this
    "attention_probs_dropout_prob": 0.1,
    "directionality": "bidi",
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 512,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pooler_fc_size": 768,
    "pooler_num_attention_heads": 12,
    "pooler_num_fc_layers": 3,
    "pooler_size_per_head": 128,
    "pooler_type": "first_token_transform",
    "type_vocab_size": 2,
    # type_vocab_size = 16,
    "vocab_size": -1
}

if I_AM_POOR :
    BERT_BASE_CONFIG["hidden_size"] = 192
    BERT_BASE_CONFIG["pooler_fc_size"]= 192
