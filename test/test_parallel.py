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

dmgr.builder.build_all(["PARALLEL_KO_KO"], config)
