def tokenize(line):
    """
    :param paragraph: str
    :return: {"text":[str, str, ...], "index":[int, int, ...], "dictionary":dict(str,int)}
    """
    return {"text": line.split(),
            "index": None,
            "dictionary": None}


def detokenize(tokens):
    return [' '.join(token) for token in tokens["text"]]
