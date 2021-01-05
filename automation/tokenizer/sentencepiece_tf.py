# https://github.com/tensorflow/text/blob/master/docs/api_docs/python/text
# https://github.com/tensorflow/text/blob/master/docs/api_docs/python/text/SentencepieceTokenizer.md

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

import tensorflow_text as text

text.SentencepieceTokenizer.detokenize(
    input, name=None
)