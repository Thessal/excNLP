import dmgr.builder
import tokenizer.sentencepiece
import embedder.bert
import ner.bert_ner
import reporter.reporter as reporter

config = {}

# Basic module load
config = tokenizer.sentencepiece.initialize(
    model_path="data/models/sentencepiece/",
    train_text_files=dmgr.builder.list_source_files(dmgrs=['TEXT_BOOK', 'TEXT_BOOK_LOW', 'TEXT_WEB']),
    sample_count=30000000, # Total sentence count in my corpus is about 80M(80000000)
    config=config,
)

# Dataset build (using modules)
dmgr.builder.build_all(["TEXT_BOOK", "TEXT_BOOK_LOW", "TEXT_WEB"], config)
dmgr.builder.build_all(["TEXT_BERT"], config)
dmgr.builder.build_all(["NER"], config)

# Module train
config = embedder.bert.initialize(model_path="data/models/bert",
                                  train_dataset="TEXT_BERT",
                                  config=config)


config = ner.bert_ner.initialize(model_path="data/models/bert_ner",
                                  train_dataset="NER",
                                  config=config) # depends on bert module

print(config["ner"]["bert_ner"]["train_loss_history"])
print(ner.bert_ner.recognize("아 침착맨이랑 섹스하고 싶다", config))
print(ner.bert_ner.recognize("안녕하세요, AI PLUS 테크블로그입니다. 오늘은 NER의 개념을 살펴보고, 포털에서 이를 어떻게 구현할 수 있을지 알아보려고 합니다.", config))
exit(0)

# tf.keras.utils.plot_model(config["embedder"]["bert"]["model"], show_shapes=True, dpi=48)

from document.document import Document

doc = Document("data/datasets/TEXT_BOOK.json", config=config)
reporter.report_to_file(doc)
