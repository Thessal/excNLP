from document.document import Document
from .formatter import heuristic as formatter
from .tokenizer import sentencepiece as tokenizer

def generate_report(text_file):
    doc = Document(formatter.format ,tokenizer.tokenize, tokenizer.detokenize)



# def _raw_lines(self):
#     """
#     Generates sentences from files
#     :return: word generator
#     """
#     self.exhausted = False
#     i = 0
#     for file in self.files:
#         i = i + 1
#         print(f"Reading file {i} out of {len(self.files)}")
#         with open(file, "r", encoding="utf-8") as fp:
#             for line in fp.readlines():
#                 for sent in self.sbd(line):
#                     yield self.explode(sent)
#     self.exhausted = True
#     print(f"{len(self.files)} files processed.")


# def iterate_line_over_file(files):
#     for file in files:
#         doc = Document(file=file)
#         while True:
#             sentence = doc._generate(unit="tokens", detail=False)
#             if doc.exhausted : break
#             yield sentence["text"]
#     yield ''
#lines = iterate_line_over_file(train_text_files)
