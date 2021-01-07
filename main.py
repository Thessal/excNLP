import dmgr.builder
import tokenizer.sentencepiece
import reporter.reporter as reporter

config = {
    "tokenizer":{
        "SentencePiece": tokenizer.sentencepiece.initialize(
            model_path="data/models/sentencepiece/",
            train_text_files=[])["tokenizer"]["SentencePiece"]
    }
}
dmgr.builder.build_all(config)

from document.document import Document
doc = Document("data/datasets/TEXT_BOOK.json", config=config)
reporter.report_to_file(doc)