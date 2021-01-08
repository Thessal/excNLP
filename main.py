import dmgr.builder
import tokenizer.sentencepiece
# import embedder.bert
import reporter.reporter as reporter

config = {}
config = tokenizer.sentencepiece.initialize(model_path="data/models/sentencepiece/", train_text_files=[], config=config)
dmgr.builder.build_all(["TEXT_BOOK", "TEXT_BERT"],config)

# this module uses derived dataset
# config = embedder.bert.initialize(model_path="data/models/bert", train_datasets=[], config=config)

from document.document import Document

if False:
    doc = Document("data/datasets/TEXT_BOOK.json", config=config)
    reporter.report_to_file(doc)
