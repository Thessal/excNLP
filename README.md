## NLP toolkit

Korean language intelligence tool with NLP.

I began this project to read books faster.  
Then I got to know nice projects like recon-ng or opensemanticsearch.

### Target Functionality
* Improve Korean (or Parapharsing, or OCR improvement, or Text summarization)
* Semantic search (or deduction, or QA training, or Translation)
* Named entity graph (NER, visualization)
* Web interface

#### Process
 * Crawl, Buy book, Record TV, Shortwave, Conversation
 * Parsing, OCR, Voice recognition, Signal processing
 * Summarize, NER
 * Query
 
### Features
* Korean specific morphological analysis based (Johab)
* Automatic paragraph segmentation & Summarization
  * Heuristic
  * Topic-based (TF-IDF & Kernel density estimation) 

### Current progress

#### Segmentation
![Paragaraph segmentation based on topic](reports/의료윤리.png)

#### Bert model
Currently training with trimmed model size.
(hidden_size = 192, 10hr training with GTX960 4GB)

Need validation and shiny plots.

```
[Evaluation result]
global_step = 100000
loss = 10.267181
masked_lm_accuracy = 0.045719843
masked_lm_loss = 9.683972
next_sentence_accuracy = 0.76
next_sentence_loss = 0.55322576
```

### TODO 
* NER
* Visualization
* Coq
