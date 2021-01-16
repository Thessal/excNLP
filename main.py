import dmgr.builder
import tokenizer.sentencepiece
import embedder.bert
import ner.bert_ner
import reporter.reporter as reporter
from document.document import Document

config = {}

# Basic module load
print("Loading module")
config = tokenizer.sentencepiece.initialize(
    model_path="data/models/sentencepiece/",
    train_text_files=dmgr.builder.list_source_files(dmgrs=['TEXT_BOOK', 'TEXT_BOOK_LOW', 'TEXT_WEB']),
    sample_count=30000000, # Total sentence count in my corpus is about 80M(80000000)
    config=config,
)

# Dataset build (using modules)
print("Building dataset")
dmgr.builder.build_all(["TEXT_BOOK", "TEXT_BOOK_LOW", "TEXT_WEB", "TEXT_WIKI", "TEXT_NAMUWIKI", "TEXT_NEWS_COMMENT"], config)
dmgr.builder.build_all(["TEXT_BERT"], config)
dmgr.builder.build_all(["NER"], config)

# Module train / load
config = embedder.bert.initialize(model_path="data/models/bert",
                                  train_dataset="TEXT_BERT",
                                  config=config)

config = ner.bert_ner.initialize(model_path="data/models/bert_ner",
                                  train_dataset="NER",
                                  config=config) # depends on bert module

# Generate report
doc = Document("data/datasets/TEXT_BOOK.json", config=config)
reporter.report_to_file(doc)




