import dmgr.builder
import tokenizer.sentencepiece
# import embedder.bert
import reporter.reporter as reporter

text_file_patterns = [pattern for dataset in ['TEXT_BOOK', 'TEXT_BOOK_LOW'] for pattern in dmgr.builder.read_json(f"data/datasets/{dataset}.json")["source_files"]]
text_files = [path for pattern in text_file_patterns for path in dmgr.builder.glob.glob(pattern)]
text_files = list(set(text_files))

config = {}
config = tokenizer.sentencepiece.initialize(
    model_path="data/models/sentencepiece/",
    train_text_files=text_files,
    config=config)

raise ValueError
dmgr.builder.build_all(["TEXT_BOOK", "TEXT_BOOK_LOW", "TEXT_BERT"],config)

# this module uses derived dataset
# config = embedder.bert.initialize(model_path="data/models/bert", train_datasets=[], config=config)

from document.document import Document

doc = Document("data/datasets/TEXT_BOOK.json", config=config)
reporter.report_to_file(doc)
