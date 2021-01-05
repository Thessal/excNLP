from document.document import Document
from .formatter import heuristic as formatter
from .tokenizer import sentencepiece as tokenizer

def generate_report(text_file):
    doc = Document(formatter.format ,tokenizer.tokenize, tokenizer.detokenize)