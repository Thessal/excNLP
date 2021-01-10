import dmgr.builder
import tokenizer.sentencepiece
import embedder.bert
import reporter.reporter as reporter

# Basic module load
config = {}
config = tokenizer.sentencepiece.initialize(
    model_path="data/models/sentencepiece/",
    train_text_files=dmgr.builder.list_source_files(dmgrs=['TEXT_BOOK', 'TEXT_BOOK_LOW']),
    sample_count=30000000, # Total sentence count in my corpus is about 80M(80000000)
    config=config,
)

# Dataset build (using modules)
dmgr.builder.build_all(["TEXT_BOOK", "TEXT_BOOK_LOW", "TEXT_BERT"], config)

# Module train
config = embedder.bert.initialize(model_path="data/models/bert",
                                  train_dataset="TEXT_BERT",
                                  config=config)

from document.document import Document

doc = Document("data/datasets/TEXT_BOOK.json", config=config)
reporter.report_to_file(doc)
