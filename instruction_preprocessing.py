import pandas as pd
import numpy as np
import re
import json

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

if len(df.columns) >= 3:
    a = input("첫 번째 column 이름을 입력하세요 : ")
    b = input("두 번째 column 이름을 입력하세요 : ")
    c = input("세 번째 column 이름을 입력하세요 : ")
    
    df = df[[a, b, c]]
    df.columns = ['instruction', 'output', 'input']
else:
    d = input("필요한 column 이름을 하나 입력하세요 : ")
    f = input("첫 번째 구분자를 입력하세요 (예: '### Human' or 'from':'human'의 human): ")
    g = input("두 번째 구분자를 입력하세요 (예: '### Assistant' or 'from':'gpt'의 gpt): ")

    human = []
    gpt = []

    for _, row in df.iterrows():
        text = row[d]
        
        if isinstance(text, dict):
            if text['from'] == f:
                human.append(text['value'])
            elif text['from'] == g:
                gpt.append(text['value'])
            '''    
            if text.get('from') == f:
                human.append(text.get('value', np.nan))
            elif text.get('from') == g:
                gpt.append(text.get('value', np.nan))
            ''' 
        else:
            split_text = re.split(re.escape(f) + '|' + re.escape(g), text)
            if len(split_text) >= 3:
                human_message = split_text[1].strip()
                gpt_message = split_text[2].strip()
                human.append(human_message)
                gpt.append(gpt_message)
            else:
                human.append(np.nan)
                gpt.append(np.nan)

    df = pd.DataFrame({
        'instruction': human,
        'output': gpt,
        'input': [np.nan] * len(human)
    })

output_file_path = input("output이 될 json 파일 경로를 입력해주세요 : ")
df.to_json(output_file_path, orient='records', force_ascii=False)
print(f"{output_file_path}")