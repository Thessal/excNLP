import sentencepiece as spm
from document.document import Document
import os
from .explode import explode
from .explode import assemble
import collections
from embedder.bert.tokenization import convert_to_unicode


def initialize(model_path, train_text_files, delete_previous_file=False,
               chunksize=1000, character_coverage=0.9995, vocab_size=200000,
               config={}):
    """
    Loads sentencepiece tokenizer. Trains if model does not exist.
    :param model_path: model directory
    :param train_text_files: one-sentence-per-line raw corpus file.
    :param delete_previous_file:
    :param chunksize:
    :param character_coverage:
    :param vocab_size:
    :return: SentencePieceProcessor
    """

    model_prefix = os.path.join(model_path, "sentencepiece")
    path_model = model_prefix + ".model"
    path_vocab = model_prefix + ".vocab"

    if delete_previous_file and os.path.isfile(path_vocab):
        if input(f"Delete {model_prefix}.[model,vocab]? [y/n]") == 'y':
            try:
                os.remove(path_model)
                os.remove(path_vocab)
            except OSError:
                print("Failed to delete.")

    if not os.path.isfile(model_prefix + ".model"):
        def iterate_line_over_file(files):
            buffer = []
            for file in files:
                with open(file, 'r') as fp:
                    for line in fp.readlines():
                        buffer.append(line)
                        if len(buffer) >= chunksize:
                            yield buffer
                            buffer = []
            yield buffer

        for chunk in iterate_line_over_file(train_text_files):
            # other args : training_sentence_size, mining_sentence_size, seed_sentencepiece_size=1000000, shrinking_factor=0.75
            # input : one-sentence-per-line raw corpus file. No need to run tokenizer, normalizer or preprocessor.
            # input_sentence_size=chunksize, shuffle_input_sentence=False,
            spm.SentencePieceTrainer.Train(sentence_iterator=chunk, model_prefix=model_prefix,
                                           character_coverage=character_coverage, vocab_size=vocab_size,
                                           model_type='unigram', )

    try:
        sp = spm.SentencePieceProcessor(
            model_file=path_model,
            model_proto=None,
            out_type=int,
            add_bos=False,
            add_eos=False,
            reverse=False,
            enable_sampling=False,
            nbest_size=-1,
            alpha=0.1
        )
        config["tokenizer_type"] = "SentencePiece"
        config["tokenizer_data"] = sp
        config["tokenizer_path"] = model_path
        with open(path_vocab, "r") as fp:
            vocab = [x.strip().split('\t') for x in fp.readlines()]
        config["tokenizer_vocab"] = {"tokens": [x[0] for x in vocab],
                                     "special": {
                                         "UNKNOWN": {"idx": 0, "token": '< unk >'},
                                         "BEGIN": {"idx": 1, "token": '< s >'},
                                         "END": {"idx": 2, "token": '< / s >'},
                                     }}
        return config
    except:
        print("Warning : Failed to load model")


def tokenize(line, mark_unk=False, config={}):
    """
    NOTE : sentencepiece automatically assembles jamo partially
    :param config: {"sp":SentencePieceProcessor}
    :param line: str
    :param mark_unk: replace original token to UNK
    :return: {"text":[str, str, ...], "index":[int, int, ...], "dictionary":dict(str,int)}
    """

    text = explode(line)
    if config["tokenizer_type"] != "SentencePiece":
        raise ValueError("Tokenizer sentencepiece is not initialized")
    sp = config["tokenizer_data"]

    if mark_unk:
        result = sp.encode(text, out_type=int)
        return {"text": [sp.IdToPiece(y) for y in result],
                "index": result,
                "dictionary": ValueError}
    else:
        result = sp.encode(text, out_type=str)
        return {"text": result,
                "index": [sp.PieceToId(y) for y in result],
                "dictionary": ValueError}


def detokenize(tokens, config={}):
    return ''.join([(' ' if ord(token[0]) == 9601 else '') + assemble(explode(token, allow_nonunique_assemble=True))
                    for token in tokens["text"]])
