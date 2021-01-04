def segmenter():
    def document_to_paragraphs(lines):
        idx = [0]+[i for i,x in enumerate(lines) is not x.strip()]+[len(lines)]
        return [lines[a,b] for a,b in zip(idx[:-1],idx[1:]) if a<b]
    def paragraph_to_sentences(paragraph):
        return paragraph.split(". ")
    def sentence_to_words(sentence):
        return sentence.split(" ")
    return {"document_to_paragraphs": document_to_paragraphs,
            "paragraph_to_sentences": paragraph_to_sentences,
            "sentence_to_words" : sentence_to_words,}