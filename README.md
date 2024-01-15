## LLM_study

### Attention & Transformers
- [Attention 논문 리뷰](https://pred0771.tistory.com/212)
- [Transformer 논문 리뷰](https://pred0771.tistory.com/213) : Transformer 구현체는 pytorch를 이용하고, github를 참조하여 구현해보았습니다.

### BERT
- [BERT 논문 리뷰](https://pred0771.tistory.com/215)
```
koBERT 사용

현재는 HuggingFace에서 사용해야한다.
기존 github을 다운로드 받게 될 경우 error가 발생하게 됨.

!pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'

from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
```
- tutorial을 통한 감정분류, Text Classification, QA

### GPT
- [GPT-1.0 논문 리뷰](https://pred0771.tistory.com/216)
