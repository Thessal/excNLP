import dmgr.builder
import tokenizer.sentencepiece
# import embedder.bert
import reporter.reporter as reporter

config = {}
config = tokenizer.sentencepiece.initialize(
    model_path="data/models/sentencepiece/",
    train_text_files=dmgr.builder.list_source_files(dmgrs=['TEXT_BOOK', 'TEXT_BOOK_LOW']),
    chunksize=1000000,
    config=config,
)

raise ValueError
dmgr.builder.build_all(["TEXT_BOOK", "TEXT_BOOK_LOW", "TEXT_BERT"], config)

# this module uses derived dataset
# config = embedder.bert.initialize(model_path="data/models/bert", train_datasets=[], config=config)

from document.document import Document

doc = Document("data/datasets/TEXT_BOOK.json", config=config)
reporter.report_to_file(doc)
