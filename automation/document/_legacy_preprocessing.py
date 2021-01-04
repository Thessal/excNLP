@staticmethod
def remove_unknown(text):
    return ''.join([x for x in text if (
            ord('가') <= ord(x) <= ord('힣') or (x in ['.', ' ', '\n'])
    )])


@staticmethod
def legacy_preprocess(_text):
    def _whitespace(_text):
        t = _text.replace('\n', ' ')
        t = ' '.join(t.split())
        return t

    def _unknown_characters(_text):
        t = _text.replace('/', ' ')
        return t

    # _parse_page()
    t = _whitespace(_text)
    t = _unknown_characters(t)
    t = t.lower()
    return t
