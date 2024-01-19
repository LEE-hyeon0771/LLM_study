import pandas as pd
import numpy as np
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("file_path", type=str, help="파일 경로를 입력하세요.")
parser.add_argument("args", nargs='*', help="추가 인자들 (컬럼 이름 또는 구분자)")
parser.add_argument("output_file_path", type=str, help="json 파일의 output_file_path")

args = parser.parse_args()

file_type = args.file_path.split('.')[-1]
if file_type == 'jsonl':
    df = pd.read_json(args.file_path, lines=True)
elif file_type == 'json':
    df = pd.read_json(args.file_path)
elif file_type == 'parquet':
    df = pd.read_parquet(args.file_path)
else:
    raise ValueError("파일 형식이 올바르지 않습니다.")

# 1. instruction, input, output과 같은 column으로 이미 셋팅된 data
if len(df.columns) >= 3:
    df = df[args.args]
    df.columns = ['instruction', 'output', 'input']

# 2. 한 column 내에 구분자로 구분되어야 하는 경우
else:
    target_column, first_delimiter, second_delimiter = args.args[0], args.args[1], args.args[2]

    instructions = []
    outputs = []
    
    for _, row in df.iterrows():
        conversation = row[target_column]
        human_msg = np.nan
        gpt_msg = np.nan

        # 1) key-value 쌍이 이미 존재하는 경우(from, value) -> (ex) 'from' : 'human', 'from' : 'gpt')
        if isinstance(conversation, list):
            for message in conversation:
                if message.get('from') == first_delimiter:
                    human_msg = message.get('value')
                elif message.get('from') == second_delimiter:
                    gpt_msg = message.get('value')
        
        # 2) key-value 쌍이 없고, 단순히 구분자로 구분되어야 하는 경우 (ex) ### Human ~~~ ### Assistant)
        elif isinstance(conversation, str):
            split_text = re.split(re.escape(first_delimiter) + '|' + re.escape(second_delimiter), conversation)
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


output_file_path = args.output_file_path
df.to_json(output_file_path, orient='records', force_ascii=False)
print(f"Output file saved to {output_file_path}")