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


- Instruction tuning 과정

1. instruction dataset이 json 파일이라면, with open등으로 json 파일을 받아와서 데이터프레임 형식으로 변환한다.
2. 사용할 토크나이저와 모델을 만든다.

```
transformers 기반의 AutoTokenizer를 사용하고, pretrained된 모델을 넣어서 토크나이징
instruction tuning시에는 AutoModelForSeq2SeqLM을 사용하여 모델을 만들어 주는것이 일반적이므로

model_name = "bert-base-uncased"  # 예시 모델 이름
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```

3. 데이터셋은 사용자가 넣은 데이터셋에서 instruction, input, output을 나누어서 CustomDataSet class를 구축해주고, 인코딩 작업을 실시한다.

``` 
 # 인코딩
        instruction_encoding = self.tokenizer(
            instruction_text, max_length=self.max_length, padding='max_length', truncation=True
        )
        input_encoding = self.tokenizer(
            input_text, max_length=self.max_length, padding='max_length', truncation=True
        )
	output_encoding = self.tokenizer(
            output_text, max_length=self.max_length, padding='max_length', truncation=True
        )
```

4. 인코딩 작업을 위에서 만들어 준 tokenizer를 사용하여 작성해준 후에 TrainingArguments와 Trainer를 설정해준다.

```
# 훈련 데이터셋 준비
train_dataset = CustomDataset(tokenizer, df)

# 훈련 설정
training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=3,              
    per_device_train_batch_size=16,  
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir='./logs',            
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# 훈련 시작
trainer.train()

이렇게 trainer를 사용해서 train 해주면, instruction tuning 된 model이 만들어지게 된다.

```

- Self-Instruct의 간단한 과정 요약
1. 모델과 토크나이저를 로드하고 평가한다.
```
model_name = "bert-base-uncased"  # 예시 모델 이름
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()
```

2. Self-Instruct 프로세스를 instruction과 대답을 생성하는 부분으로 나누어 구현해 준다.
```
def generate_instruction(model, tokenizer, seed_sentence, max_length=50):
    inputs = tokenizer.encode(seed_sentence, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def model_response_to_instruction(model, tokenizer, instruction, max_length=50):
    inputs = tokenizer.encode(instruction, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

3. seed 문장이나, seed 파일을 불러오고, 반복 수 만큼 무작위로 선택된 시드를 기반으로 새로운 instruction과 response를 생성하여 model이 다양한 유형의 instruction을 생성할 수 있도록 한다.
```
seed_sentences = 시드문장
for _ in range(5):
    seed = random.choice(seed_sentences)
    instruction = generate_instruction(model, tokenizer, seed)
    response = model_response_to_instruction(model, tokenizer, instruction)
    print(f"Instruction: {instruction}\nResponse: {response}\n")
```
