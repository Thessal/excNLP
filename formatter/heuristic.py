import re


def format(lines, config={}):
    paragraphs = _segment(lines, debug=False)
    return [[_filter(x) for sentence in paragraph for x in sbd(sentence)] for paragraph in paragraphs]


_boundary = re.compile(r'[!\?\.] ')


def sbd(text, sentence_length_min=20):  # sentence boundary disambiguation
    sep = [0] + [x.end(0) for x in _boundary.finditer(text)] + [len(text)]
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


_filter_separator = re.compile(r'^([-=#]{3}).*\1$')
_filter_citation = re.compile(r'(?<=[^.\-+*/)( 0-9])[0-9]+')


def _filter(line):
    # re.sub(_filter_separator)
    line = line.strip()
    line = _filter_separator.sub(' ', line, count=1)
    line = _filter_citation.sub('', line, count=3)
    return line


def _calc_stat(lines, debug=False):
    # determine document type
    stat_len = len(lines)
    stat_chars = sum([len(x) for x in lines])
    stat_empty_mask = [(not bool(x.strip())) for x in lines]
    stat_empty = sum(stat_empty_mask)
    ## Stats for empty line seperated
    stat_segment_mask = [0] + [i for i, x in enumerate(stat_empty_mask) if x] + [len(lines)]
    stat_segment_mask = zip(stat_segment_mask[:-1], stat_segment_mask[1:])
    stat_segment_length = [sum([len(x) for x in lines[a:b]]) for a, b in stat_segment_mask]
    stat_segment_length = [x for x in stat_segment_length if x > 0]
    stat_segment_size_avg = sum(stat_segment_length) / len(stat_segment_length)
    stat_segment_size_max = max(stat_segment_length)
    ## Stats for newline seperated
    stat_noperiod = sum([len(x) for x in lines if not ('.' in x[-3:])]) / stat_chars  # .
    stat_long = sum([len(x) for x in lines if (x.count('.') > 1 and len(x) > 100)]) / stat_chars
    stat_short = sum([len(x) <= 100 for x in lines]) / stat_len  # unused
    stat_veryshort = sum([(len(x) < 10) for x in lines]) / stat_len  # unused

    format_split = stat_noperiod > 0.35  # fixed width line break, need to be joined
    format_exploded = (stat_long <= 0.01) and (
            stat_segment_size_avg <= 100)  # lots of empty line, need to replace '\n\n' into \n,separated by multiple veryshort lines
    format_para = (stat_long > 0.35 or stat_segment_size_avg > 2000)  # Paragraph per line
    format_sent = stat_long <= 0.35  # Sentence per line, separated by empty line
    if debug:
        print(f"stat_veryshort:{stat_veryshort:.2f}\n"
              f"stat_short:{stat_short:.2f}\n"
              f"stat_long:{stat_long:.2f}\n"
              f"lines does not end with period:{stat_noperiod:.2f}\n"
              f"avg char per empty-line segment:{stat_segment_size_avg:.2f}\n"
              f"max char per empty-line segment:{stat_segment_size_max:.2f}"
              )
        if format_split: print("split")
        if format_exploded: print("explode")
        if format_para: print("paragraph")
        if format_sent: print("sentence")
        print()
    return dict(segment_size_avg=stat_segment_size_avg, segment_size_max=stat_segment_size_avg,
                noperiod=stat_noperiod, long=stat_long, short=stat_short, veryshort=stat_veryshort,
                format_split=format_split, format_exploded=format_exploded, format_para=format_para,
                format_sent=format_sent)


def _split_paragraph(text):
    output = [x + '.' for x in text.split('. ')]
    output[-1] = output[-1][:-1]
    return output


def _segment(lines, debug=False):
    """
    Heuristics for paragraph segmentation.
    :param lines: fp.readlines()
    :param debug: prints some results
    :return: [ [line,line,...], [line,line,...], ... ] (approximately)
    """
    stat = _calc_stat(lines, debug)
    if stat["format_exploded"]:
        lines = ('\n'.join(lines).replace("\n\n", "\n")).split('\n')
        lines = [v for i, v in enumerate(lines) if i == 0 or v != lines[i - 1]]
    if stat["format_split"]:
        lines = ' '.join([(x if len(x) > 3 else x + '\n') for x in lines]).split('\n')

    stat = _calc_stat(lines, debug) if (stat["format_split"] or stat["format_exploded"]) else stat
    if stat["format_para"]:
        # lines = [x for y in [x.split(". ") for x in lines] for x in [x+'.' for x in y[:-1]]+[y[-1]]]
        lines = [x for y in [_split_paragraph(x) for x in lines] for x in y]
    elif stat["format_sent"]:
        pass
    else:
        pass

    is_valid = (stat['segment_size_avg'] < 2000)  # segmentation result is valid
    if not is_valid:
        print(f"Paragraph too large ({stat['segment_size_avg']})")
        N = 10
        lines = [y for x in [lines[i * N:i * N + N] + [''] for i in range(int(len(lines) / N))] for y in x]

    output = [v.strip() for i, v in enumerate(lines) if i == 0 or (len(lines[i]) + len(lines[i - 1])) > 3]
    if debug:
        print('\n'.join(output[:50]))

    # Group by paragraph
    paragraph_boundary = [i for i,x in enumerate(output) if not x]
    output = [output[a:b] for a,b in zip([0]+[x+1 for x in paragraph_boundary],paragraph_boundary+[len(output)])]
    return output
