## MEMO

### Study
Entropy 
 * Intrinsic dimension : F Camastra 2002
 * Perplexity (evaluation metric, T-SNE)

Graph
 * Knowledge graph
 * Trie : radix tree 비슷한 개념임. 
   * array의 최대값을 찾는데 각 element가 복잡한 수식이라서 접근하기가 느릴 때 어떻게 빠르게 계산할 수 있을까.
   [a,b,c,d,e,f] -> [ [max(a,b,c):a,b,c], [max(d,e,f):d,e,f] ] 이렇게 바꾸면 더 빠르다.
   이게 NLP와 alpha search등에 활용될 수 있을 것 같다.
   그런데 이걸 functional 하게 만드는게 간단하지 않다. https://github.com/qntm/greenery 써보면 무척 느리다.
   * 예제들
     * NLP 동의어,시제,어미 등 처리할때 :
     [먹다, 먹음, 먹었음, 잠, 자다, 자기] -> [ [먹-:먹다,먹음,먹었음], [자-:잠,자다,자기] ]
     * alpha template(혹은 regex등의 automata)의 parameter 최적화 할때 :
     {ts_max({ts1|ts2|ts3},{10|20|30})} = [ ts_max(ts1,10), ts_max(ts1,20), ... ]
     -> [ [ts1:ts_max(ts1,10), ...], [ts2:ts_max(ts2,10), ...] ]
     * coq에서도 같은 문제를 겪는다. https://softwarefoundations.cis.upenn.edu/vfa-current/Trie.html#loop:4
     IOMonad를 쓰느니 그냥 C를 쓰지. 나도 그래서 걍 C 파서를 만들어서 따로 묶거나 했다.
     이걸 바꿔서 좀 써먹을수 있지 않을까 싶다. 
   * 확장 
     * BST의 한계는 1차원이라는 거다. BST를 이용한 상수 시간의 자연수 비교는 edit distance의 1차원 근사다.
     sierpinski diagram을 생각하면 edit distance를 구할 수 있다. 1.6 차원이라고 하니깐 대략 node 하나마다 1.6개의 neighbor를 탐색하면 될것이다.
     * NLP나 alpha template parameter optimization에서는, node 개수가 일정하지 않다.
     그래서 local dimension을 정의해야 한다. Hausdorff dimension을 이용하면 가능하다.
     * 그런데 이걸 빠르게 계산할 수 있을까?
     http://math.bme.hu/~morap/sierpinski.pdf
   * 걍 expand 해서 MCMC 돌리는게 나은거 같기도 하고...

LSTM
 * POS tagger
 * BERT, NER
 
```
How to visualize TF-Lite inference graph?

TensorFlow Lite models can be visualized using the visualize.py script in the TensorFlow Lite repository. You just need to:

Clone the TensorFlow repository
Run the visualize.py script with bazel:

  bazel run //tensorflow/lite/tools:visualize \
       /Users/jongkook90/Projects/book-report/automation/proprietary/model.tflite \
       /Users/jongkook90/Projects/book-report/automation/proprietary/visualized_model.html
  bazel run //tensorflow/lite/tools:visualize \
       /Users/jongkook90/Projects/tflite/python/examples/classification/models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite \
       /Users/jongkook90/Projects/book-report/automation/proprietary/visualized_model.html

inputs	outputs
[0]	[1413]

inputs	outputs
[0]	[1]
```
Graph 최적화에 문제가 있다. 원인 분석 필요.

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

