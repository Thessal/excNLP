## MEMO

### Study
Entropy 
 * Intrinsic dimension : F Camastra 2002
 * Perplexity (evaluation metric, T-SNE)

Graph
 * Knowledge graph
 * Trie in coq

LSTM
 * POS tagger
 * BERT, NER

### TODO
 * inter-sentence summarization 
   * Sentence pooling method
 * intra-sentence summarization
   * attention mechanism
   * Paraphrasing
   * translation, gpt3 Corrected English

### One-liners
Attentions (https://wikidocs.net/book/2155)
 * TF-IDF : semantic attention, normalized by Poisson model, exponential cdf (l=l_0-exp(p))
 * Attention : sentence(LSTM seq) syntactic info, DL fitting
 * Topic modeling : clustering (SVD, LDA)
   * LDA - 베타분포를 prior로 topic probability 계산
     * 베타분포 - conjugacy가 있어서 log likelihood 계산하기 좋다
     * posterior : prior와 measured를 hyperparameter로 vector convolution, 내적을 kl-divergence로 계산하여 weight 준다. 

확률분포 구글링좀 그만하자
 * 확률분포간-관계도 : https://losskatsu.github.io/statistics/betadist/#참고-확률분포간-관계도
   * 정규분포, 포아송분포, 이항분포 : 근사(a>>1 b>>1, a<<b, 이산)
   * 이항분포, 초기하분포 : 복원추출, 비복원추출
   * 이항분포, 음이항분포 : 성공횟수, 성공할때까지의 시도횟수
   * 음이항분포, 기하분포 : r번째, 첫번째
   * 기하분포, 지수분포 : 1st발생 시행횟수, 1st발생 대기시간
   * 지수분포, 감마분포 : 1st발생 대기시간, r번째 발생 대기시간
   * 지수분포, 포아송분포 : 1st발생 대기시간, 단위시간당 성공횟수
   * 감마분포, 카이제곱분포 : 일반화, 정규분포 분산
