import sentencepiece as spm
from document.document import Document
import os
from .explode import explode
from .explode import assemble


def initialize(model_path, train_text_files, delete_previous_file=False,
               chunksize=1000, character_coverage=0.9995, vocab_size=200000):
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
    path_model = model_prefix+".model"
    path_vocab = model_prefix+".vocab"

    if delete_previous_file and os.path.isfile(path_vocab):
        if input(f"Delete {model_prefix}.[model,vocab]? [y/n]") == 'y':
            try:
                os.remove(path_model)
                os.remove(path_vocab)
            except OSError:
                print("Failed to delete.")

    if not os.path.isfile(model_prefix+".model"):
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
    except:
        print("Warning : Failed to load model")

    return sp


def tokenize(text, sp, mark_unk=False):
    """
    NOTE : sentencepiece automatically assembles jamo partially
    :param sp: SentencePieceProcessor
    :param text: str
    :param mark_unk: replace original token to UNK
    :return: {"text":[str, str, ...], "index":[int, int, ...], "dictionary":dict(str,int)}
    """
    # raise ValueError("Tokenizer sentencepiece is not initialized")

    text = explode(text)

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


def detokenize(tokens):
    return ''.join([(' ' if ord(token[0]) == 9601 else '') + assemble(explode(token, allow_nonunique_assemble=True))
                    for token in tokens["text"]])
