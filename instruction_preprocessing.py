import pandas as pd
import numpy as np
import re

file_path = input("파일 경로를 입력하세요 : ")
file_type = file_path.split('.')[-1]

if file_type == 'jsonl':
    df = pd.read_json(file_path, lines=True)
elif file_type == 'json':
    df = pd.read_json(file_path)
elif file_type == 'parquet':
    df = pd.read_parquet(file_path)
else:
    raise ValueError("파일 형식이 올바르지 않습니다.")

# 1. instruction, input, output과 같은 column으로 이미 셋팅된 data
if len(df.columns) >= 3:
    a = input("첫 번째 column 이름을 입력하세요 : ")
    b = input("두 번째 column 이름을 입력하세요 : ")
    c = input("세 번째 column 이름을 입력하세요 : ")
    
    df = df[[a, b, c]]
    df.columns = ['instruction', 'output', 'input']

# 2. 한 column 내에 구분자로 구분되어야 하는 경우
else:
    d = input("필요한 column 이름을 하나 입력하세요 : ")
    f = input("첫 번째 구분자를 입력하세요 (예: '### Human' or 'from':'human'의 human): ")
    g = input("두 번째 구분자를 입력하세요 (예: '### Assistant' or 'from':'gpt'의 gpt): ")

    instructions = []
    outputs = []

    for _, row in df.iterrows():
        conversation = row[d]
        human_msg = np.nan
        gpt_msg = np.nan

        # 1) key-value 쌍이 이미 존재하는 경우(from, value) -> (ex) 'from' : 'human', 'from' : 'gpt')
        if isinstance(conversation, list):
            for message in conversation:
                if message.get('from') == f:
                    human_msg = message.get('value')
                elif message.get('from') == g:
                    gpt_msg = message.get('value')
        
        # 2) key-value 쌍이 없고, 단순히 구분자로 구분되어야 하는 경우 (ex) ### Human ~~~ ### Assistant)
        elif isinstance(conversation, str):
            split_text = re.split(re.escape(f) + '|' + re.escape(g), conversation)
            if len(split_text) >= 3:
                human_msg = split_text[1].strip()
                gpt_msg = split_text[2].strip()

        instructions.append(human_msg)
        outputs.append(gpt_msg)

    df = pd.DataFrame({
        'instruction': instructions,
        'output': outputs,
        'input': [np.nan] * len(instructions)
    })

print(df.columns)
print(df.head())

output_file_path = input("output이 될 json 파일 경로를 입력해주세요 : ")
df.to_json(output_file_path, orient='records', force_ascii=False)
print(f"{output_file_path}")