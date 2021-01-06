from document.document import TextIO
import os
from .trie import Trie

## Tokenizer endcoder related
import pickle
import glob


class Document(TextIO):
    """
    Generate results for each files
    """

    def stats(self):
        # Document statistics
        raise NotImplementedError

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
