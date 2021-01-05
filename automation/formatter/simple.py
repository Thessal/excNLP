def format(lines):
    idx = [-1] + [i for i, x in enumerate(lines) if not x.strip()] + [len(lines)]
    document = [lines[a + 1: b] for a, b in zip(idx[:-1], idx[1:]) if a < b]
    document = [[x for sentence in paragraph for x in sentence.split(". ")] for paragraph in document]
    return document
