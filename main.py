import dmgr.builder
import formatter.simple
import formatter.heuristic
import tokenizer.simple
import tokenizer.sentencepiece

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
