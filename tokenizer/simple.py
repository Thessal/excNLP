def initialize(config= {}):
    config["tokenizer_type"] = "simple"
    return config

def tokenize(line, config={}):
    """
    :param paragraph: str
    :return: {"text":[str, str, ...], "index":[int, int, ...], "dictionary":dict(str,int)}
    """
    return {"text": line.split(),
            "index": None,
            "dictionary": None}

def detokenize(tokens, config={}):
    return ' '.join(tokens["text"])

