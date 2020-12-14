from .file_io import TextIO
import os
from trie import Trie

## Tokenizer endcoder related
import pickle
import glob
from konlpy.tag import Kkma


class Document(TextIO):

    def __init__(self, pattern, limit=None):
        """
        :param pattern: "/foo/bar/baz/wildcard.txt"
        :param limit: if specified, limits max line
        """
        super().__init__(pattern)
        replace_extension = lambda file, ext: os.path.join(os.path.dirname(file),
                                                           os.path.splitext(os.path.basename(file))[0] + '.' + ext)
        self.legacy = [
            {"sentences": [], "orig_text": [], "embeddings": [], "summary_onehot": [],
             "file_input": file,
             "file_summary": replace_extension(file, "summary.txt"),
             "file_pkl": replace_extension(file, str(limit or "all") + ".pkl"),
             "path": os.path.splitext(file)[0]
             }
            for file in list(glob.glob(self.pattern))]

    def trie(self):
        for d in self.documents():
            yield Trie(d.split(' '))

    def iter_text(self):
        """
        Global indexer
        :return: generator for (text, doc_id, paragraph_id, sentence_id, word_id)
        """
        raise NotImplementedError


def legacy_sentences_from_raw_text(path, limit=None, force=False):
    pkl_path = path + '.' + str(limit or 'all') + ".pkl"
    if glob.glob(pkl_path) and (not force):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
            tokens = data['tokens']
            text = data['text']
    else:
        print("POS tagging...")
        kkma = Kkma()
        from .preprocessing import legacy_preprocess
        with open(path, "r", encoding="utf-8") as reader:
            text = reader.readlines()
            if limit: text = text[0:limit]
            text = legacy_preprocess.__func__(''.join(text))
            text = kkma.sentences(text)
            tokens = [kkma.pos(x) for x in text]
            with open(pkl_path, "wb") as f:
                pickle.dump({'tokens': tokens, 'text': text}, f)
    return tokens, text
