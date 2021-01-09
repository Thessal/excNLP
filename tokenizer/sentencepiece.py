import sentencepiece as spm
import os
from .explode import explode
from .explode import assemble
import re

class LinesIterator:
    def __init__(self, files, chunk_size):
        self.files = iter(files)
        self.chunk_size = max(1, chunk_size)
        self.buffer = []
        self.depleted = False
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        while len(self.buffer) < self.chunk_size:
            try:
                with open(next(self.files), 'r') as fp:
                    lines = []
                    for line in fp.readlines():
                        line = line.strip()
                        if len(line) > 1000:
                            line = line.replace('\x00',' ')
                            line = ' '.join(line.split())
                            line = re.sub(r'(.)\1{5,}', r' \1 ', line)
                            sep = [m.end() for m in re.finditer(r'(다\.|\. )', line)]
                            line = [line[a:b].strip() for a,b in zip([0]+sep,sep+[len(line)])]
                            if any([len(x)>1000 for x in line]):
                                print(f"Too long : {line}")
                            lines.extend(line)
                        else:
                            lines.append(line)
                    lines = [line for line in lines if line.strip()]
                    self.buffer.extend(lines)
            except StopIteration:
                if not self.depleted:
                    self.depleted = True
                    self.buffer.extend([''] * (self.chunk_size - (self.index % self.chunk_size)))
                break
        self.index += 1
        # print(self.index, explode(self.buffer[0]))
        return explode(self.buffer.pop(0))


def initialize(model_path, train_text_files, delete_previous_file=False,
               chunksize=0, character_coverage=0.9995, vocab_size=200000,
               config={}):
    """
    Loads sentencepiece tokenizer. Trains if model does not exist.
    :param model_path: model directory
    :param train_text_files: one-sentence-per-line raw corpus file.
    :param delete_previous_file:
    :param chunksize: sentence count per batch. 0 for one shot training. use this option to save memory.
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
        # TODO : need version control. e.g. train_text update
        lines_iterator = LinesIterator(train_text_files, chunksize)
        # https://github.com/google/sentencepiece/blob/master/doc/options.md
        while not lines_iterator.depleted:
            spm.SentencePieceTrainer.Train(sentence_iterator=lines_iterator, model_prefix=model_prefix,
                                           character_coverage=character_coverage, vocab_size=vocab_size,
                                           input_sentence_size=chunksize, shuffle_input_sentence=False,
                                           model_type='unigram', train_extremely_large_corpus=bool(chunksize==0) )

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
        if "tokenizer" not in config.keys():
            config.update({"tokenizer": {"sentencepiece": {}}})
        config["tokenizer"]["sentencepiece"]["data"] = sp
        config["tokenizer"]["sentencepiece"]["path"] = model_path
        with open(path_vocab, "r") as fp:
            vocab = [x.strip().split('\t') for x in fp.readlines()]
        config["tokenizer"]["sentencepiece"]["vocab"] = {"tokens": [x[0] for x in vocab],
                                                         "special": {
                                                             "UNKNOWN": {"idx": 0, "token": '<unk>'},
                                                             "BEGIN": {"idx": 1, "token": '<s>'},
                                                             "END": {"idx": 2, "token": '</s>'},
                                                         }}
        # config["tokenizer"]["sentencepiece"]["detokenizer"] = detokenize
        return config
    except Exception as e:
        print("Warning : Failed to load model")
        raise (e)


def tokenize(line, mark_unk=False, config={}):
    """
    NOTE : sentencepiece automatically assembles jamo partially
    :param config: {"sp":SentencePieceProcessor}
    :param line: str
    :param mark_unk: replace original token to UNK. detokenization is lossy.
    :return: {"text":[str, str, ...], "index":[int, int, ...], "dictionary":dict(str,int)}
    """

    text = explode(line)
    if "sentencepiece" not in config["tokenizer"].keys():
        raise ValueError("Tokenizer sentencepiece is not initialized")
    sp = config["tokenizer"]["sentencepiece"]["data"]

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
    if not isinstance(tokens, dict):
        tokens = {"text": [str(x) for x in tokens]}
    assert isinstance(tokens, dict)
    return assemble(
        ''.join(
            [(' ' if ord(token[0]) == 9601 else '') + explode(token, allow_nonunique_assemble=True)
             for token in tokens["text"]]
        ))
    # return ''.join([(' ' if ord(token[0]) == 9601 else '') + assemble(explode(token, allow_nonunique_assemble=True))
    #                 for token in tokens["text"]])
