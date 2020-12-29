from document_model import Trie
import pickle
import numpy as np
import sentencepiece as spm
import os
import tensorflow as tf  # tf.enable_eager_execution()
import tensorflow_text as tf_text


class Tokenizer:
    from document_model.explode import explode
    from document_model.explode import assemble
    def __init__(self, working_dir, text_files, rebuild=False, postfix_sensitivity=50):
        """
        Set up tokenizer.
        :param rebuild: If rebuild, read text file and build trie for tokenizer. It takes few hours
        :param text_files: Used for rebuilding
        :param working_dir: Cache files for rebuilding / loading
        :param postfix_sensitivity: postfix detection sensitivity
        """
        self.working_dir = working_dir
        self.trie = Trie()

        cache_files = self._text_to_pkl(files=text_files, chunksize=1000, thres=100, run=rebuild)
        for cache in cache_files:
            with open(cache, 'rb') as handle:
                for c in pickle.load(handle):
                    self.trie.insert(c[0], c[1])

        self.count_all = self.trie.query('')[0][1]
        self.average_postfix_count = postfix_sensitivity

    def _raw_words(self, files):
        """
        Generates words from files
        :param files: list containing txt filenames
        :return: word generator
        """
        for file in files:
            with open(file, "r", encoding="utf-8") as fp:
                for line in fp.readlines():
                    if not line.strip(): pass
                    words = line.strip().split(' ')
                    for word in words:
                        word = word.strip()
                        if word: yield word
        print(f"{len(files)} files processed.")

    def _text_to_pkl(self, files, chunksize=1000, thres=100, run=False):
        """
        In one loop, read {chunksize} files.
        For each loop, truncate words not popular than {thres}.
        :param files: text files
        :param chunksize: generate a trie for every {chunksize} files.
        :param thres: for each trie, truncate words not popular than {thres}.
        :param run: if False, then only generate pkl paths
        :return: pkl file path
        """
        print(f"Total {len(files)} files.")
        indices = list(range(int(len(files) / chunksize)))
        pkl_files = [f"{self.working_dir}Trie_{x}.pkl" for x in indices]
        if run:
            for x in indices:
                begin = x * chunksize
                end = min(len(files), begin + chunksize)
                trie = Trie()
                for word in self._raw_words(files=files[begin:end]):
                    trie.insert(word)
                trie.output = []
                trie.dfs(trie.root, '', include_subword=True, count_recursive=False)
                data = [x for x in trie.output if x[1] > thres]
                with open(pkl_files[x], 'wb') as handle:
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return pkl_files

    def tokenize(self, text):
        """
        Tokenize input text
        :param text:
        :return:
        """
        output = []
        for x in text.strip().split(' '):
            x = x.replace('/', '')
            if not x: pass
            x_e = self.explode(x)

            # Try trie query
            prefix_length_min = 3
            queries = [x_e[:i] for i in range(prefix_length_min, len(x_e) + 1)]
            # print(x)
            # print(queries)
            match = [self.trie.query(self.assemble(x)) for x in queries]
            # print([len(x) for x in match])
            match = [x for x in match if x]
            tf = [x[0][1] / self.count_all for x in match]
            # print(tf)
            idf = [-np.log(len(x) / self.average_postfix_count) for x in match]
            # print(idf)
            tfidf = [x[0] * x[1] for x in zip(tf, idf)]
            # print(tfidf)
            prefix_length = prefix_length_min + np.argmax(tfidf) if tfidf else len(x_e)
            # print(prefix_length)
            # print(x_e[:prefix_length])
            # print(x_e[prefix_length:])
            prefix, postfix = self.assemble(x_e[:prefix_length]), self.assemble(x_e[prefix_length:])
            if prefix: output.append(prefix)
            if postfix: output.append(postfix)
        return output


# https://github.com/tensorflow/text/blob/master/docs/api_docs/python/text
class TokenizerSpm:
    from document_model.explode import explode
    from document_model.explode import assemble
    def __init__(self, wd, train_args=None):
        """
        :param files: list containing txt filenames
        :param wd: working directory
        :param files: (optional) text files for train
        """
        self.wd = wd
        self.model_prefix = os.path.join(wd, "sentencepiece")
        if train_args:
            self.files = train_args['files']
            self.character_coverage = train_args['character_coverage']
            self.vocab_size = train_args['vocab_size']
        self.sp = None
        self.exhausted = False

    @staticmethod
    def sbd(text, sentence_length_min=20):  # sentence boundary disambiguation
        ses = ['. ', '? ']  # sentence ending suffixes
        sep = [0] + [cur + 1 for cur in range(len(text) - 1) if (text[cur:cur + 2] in ses)] + [len(text)]
        sen = [text[i[0]:i[1]].strip() for i in zip(sep[:-1], sep[1:])]
        for i in range(len(sen) - 1):
            if len(sen[i]) < sentence_length_min:
                sen[i + 1] = sen[i] + ' ' + sen[i + 1]
                sen[i] = ''
        if len(sen) > 1:
            if len(sen[-1]) < sentence_length_min:
                sen[-2] = sen[-2] + ' ' + sen[-1]
                sen[-1] = ''
        sen = [x for x in sen if x]
        return sen

    def _raw_lines(self):
        """
        Generates sentences from files
        :return: word generator
        """
        self.exhausted = False
        i = 0
        for file in self.files:
            i = i + 1
            print(f"Reading file {i} out of {len(self.files)}")
            with open(file, "r", encoding="utf-8") as fp:
                for line in fp.readlines():
                    for sent in self.sbd(line):
                        yield self.explode(sent)
        self.exhausted = True
        print(f"{len(self.files)} files processed.")

    def train(self, delete_previous_file=False, chunksize=1000):
        if delete_previous_file:
            if input(f"Delete {self.model_prefix}.[model,vocab]? [y/n]") == 'y':
                try:
                    os.remove(self.model_prefix + '.model')
                    os.remove(self.model_prefix + '.vocab')
                except OSError:
                    print("Failed to delete.")
        # print(f"Total {len(self.files)} files.")
        # for i in range(0, len(self.files), chunksize):
        #     lines = self._raw_lines(self.files[i:i + chunksize])
        #     spm.SentencePieceTrainer.Train(sentence_iterator=lines, model_prefix=self.model_prefix,
        #                                    character_coverage=self.character_coverage, vocab_size=self.vocab_size)

        # for i in range(0, len(self.files), chunksize):
        #     lines = self._raw_lines(self.files[i:i + chunksize])
        lines = self._raw_lines()
        while not self.exhausted:
            spm.SentencePieceTrainer.Train(sentence_iterator=lines, model_prefix=self.model_prefix,
                                           character_coverage=self.character_coverage, vocab_size=self.vocab_size,
                                           input_sentence_size=chunksize, shuffle_input_sentence=False)
        # How to specify alphabet size...?
        # --input: one-sentence-per-line raw corpus file. No need to run tokenizer, normalizer or preprocessor. By default, SentencePiece normalizes the input with Unicode NFKC. You can pass a comma-separated list of files.
        # --model_prefix: output model name prefix. <model_name>.model and <model_name>.vocab are generated.
        # --vocab_size: vocabulary size, e.g., 8000, 16000, or 32000
        # --character_coverage: amount of characters covered by the model, good defaults are: 0.9995 for languages with rich character set like Japanse or Chinese and 1.0 for other languages with small character set.
        # --model_type: model type. Choose from unigram (default), bpe, char, or word. The input sentence must be pretokenized when using word type.
        #
        # trainer_interface.cc(112) LOG(WARNING) Too many sentences are loaded! (45581210), which may slow down training.
        # trainer_interface.cc(114) LOG(WARNING) Consider using --input_sentence_size=<size> and --shuffle_input_sentence=true.
        # trainer_interface.cc(117) LOG(WARNING) They allow to randomly sample <size> sentences from the entire corpus.
        # --input_sentence_size: The number of lines spm_train first loads. Remaining lines are simply discarded. Since spm_train loads entire corpus into memory, this size will depend on the memory size of the machine. It also affects training time.
        # --training_sentence_size: The number of lines to train BPE/unigram model.
        # --mining_sentence_size: The number of lines to generate seed pieces. spm_train extracts frequent substring from the corpus with suffix array, which requires O(10N) memory space. This setting is valid when --model_type=unigram.
        # --seed_sentencepiece_size: The size of seed pieces. This setting is valid when --model_type=unigram.
        # In general, we can use the default values except for --input_sentence_size. Training spm with more than 10M sentences requires tons of memory and CPU resources.
        #
        # spm_train loads all files/sentences by default
        # --input_sentence_size=N lets spm_train load randomly shuffled N lines.
        # --shuffle_input_sentence=false loads first N lines instead.
        # --mining_sentence_size and --training_sentence_size are deprecated.
        try:
            self.load(enable_tf=False)
        except:
            print("Warning : Failed to load model")

    def load(self, enable_tf):
        if enable_tf:
            # TF
            # model = open(f"{MODEL_PREFIX}.model", "rb").read()
            # text.SentencepieceTokenizer(
            #     model=None, out_type=dtypes.int32, nbest_size=0, alpha=1.0, reverse=False,
            #     add_bos=False, add_eos=False, name=None
            # )
            # tensorflow_sp_out_int = text.SentencepieceTokenizer(model=model)
            # tensorflow_sp_out_str = text.SentencepieceTokenizer(model=model, out_type=tf.string)
            # print(tensorflow_sp_out_str.tokenize(["토크나이저 테스트"]))
            # print(tensorflow_sp_out_int.tokenize(["토크나이저 테스트"]))
            #
            # print(tensorflow_sp_out_str.tokenize(["토크나이저 테스트"]))
            raise NotImplementedError
        else:
            # Non-TF
            self.sp = spm.SentencePieceProcessor(
                model_file=self.model_prefix + '.model',
                model_proto=None,
                out_type=int,
                add_bos=False,
                add_eos=False,
                reverse=False,
                enable_sampling=False,
                nbest_size=-1,
                alpha=0.1
            )
            # >> > sp.encode(['This is a test', 'Hello world'], out_type=int)
            # [[284, 47, 11, 4, 15, 400], [151, 88, 21, 887]]
            # >> > sp.encode('This is a test', out_type=str)
            # ['▁This', '▁is', '▁a', '▁', 't', 'est']
            # >> > sp.encode(['This is a test', 'Hello world'], out_type=str)


    def tokenize(self, text, debug=False, out_type=str, mark_unk=True):
        """

        :param text:
        :param debug:
        :param out_type: str or int
        :param mark_unk: replace original token to UNK
        :return:
        """
        if isinstance(text, str) :
            text = [text]
        text = [self.explode(x) for x in text]
        if debug:
            # print(self.sp.encode(text, out_type=str)) # TODO : prevent sentencepiece automatic assemble
            output = [
                [(' ' if ord(t[0]) == 9601 else '/') + self.assemble(self.explode(t, allow_nonunique_assemble=True)) for
                 t in x] for x in
                self.sp.encode(text, out_type=str)]
            print('\n'.join([''.join(x) for x in output]))

        if out_type==str and mark_unk :
            output = [[self.sp.IdToPiece(y) for y in x] for x in self.sp.encode(text, out_type=int)]
        else:
            output = self.sp.encode(text, out_type=out_type)
        return output[0] if len(output)==1 else output
