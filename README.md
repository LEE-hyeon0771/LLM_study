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
```
from transformers import TrainingArguments

args = TrainingArguments(
    "bert-finetuned-squad",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    push_to_hub=True,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
)
trainer.train()

위 코드들은 huggingface_hub에서 login으로 token을 받아서 사용할 수 있는 코드이다.

from huggingface_hub import notebook_login
notebook_login()

이 코드로 huggingface 상에서 token을 다운받아 로그인을 진행할 수 있다.

TrainingArguments를 실행시킬 때, ImportError: Using the Trainer with PyTorch requires accelerate>=0.20.1: Please run pip install transformers[torch] or pip install accelerate -U
이런 오류가 발생하는 경우가 있는데,
이 때 해결방법
1. Run pip install accelerate -U in a cell
2. In the top menu click Runtime → Restart Runtime
3. Do not rerun any cells with !pip install in them
4. Rerun all the other code cells and you should be good to go!

이렇게 해결이 가능하다.
```

### GPT
- [GPT-1.0 논문 리뷰](https://pred0771.tistory.com/216)


### 공개 한국어 Instructions dataset 가공하기
- [korean - instruction dataset preprocessing tutorial](https://github.com/LEE-hyeon0771/instruction_dataset/tree/main)


### Self-Instruction
- [Self-Instrution 논문 리뷰](https://pred0771.tistory.com/220)
- [Instruction tuning & Self-Instruct 과정](https://pred0771.tistory.com/222)


### LLM hyper-parameters
- [Hallucination & Inference와 연관된 LLM Hyperparameters](https://github.com/LEE-hyeon0771/LLM_study/blob/main/LLM_parameters.md)

### LLaMA
- [LLaMA 논문 리뷰](https://pred0771.tistory.com/224)

### LLM 기법
- [PEFT, LoRA, QLoRA, RAG](https://pred0771.tistory.com/225)
- [langchain & llama-index](https://github.com/LEE-hyeon0771/langchain_llama-index)

### Prompt Engineering(보고서)
- [보고서 Task](https://github.com/LEE-hyeon0771/LLM_study/blob/main/prompt%20engineering.md)


### BERT 모델 Fine-tuning
- [BERT 모델 Fine-tuning & HuggingFace Upload](https://huggingface.co/HyeoniLEE/book_finetuned_model)
- [Fine-tuning Book Dataset](https://huggingface.co/datasets/HyeoniLEE/books_dataset)


### 인턴십 최종 발표
- [최종발표 ppt](https://github.com/LEE-hyeon0771/LLM_study/blob/main/LLM_%EC%84%B8%EA%B3%84%EB%A1%9C%EC%9D%98_%EC%97%AC%EC%A0%95.pdf)

### LLM paper 리뷰
- [Are Emergent Abilities of Large Language Models a Mirage?](https://pred0771.tistory.com/237)
- [Jailbroken: How Does LLM Safety Training Fail?](https://pred0771.tistory.com/239)
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://pred0771.tistory.com/241)
- [ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings](https://pred0771.tistory.com/243)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://pred0771.tistory.com/244)
- [Scaling Data-Constrained Language Models](https://pred0771.tistory.com/245)
- [DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models](https://pred0771.tistory.com/246)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://pred0771.tistory.com/247)
