import pandas as pd
from tqdm import tqdm
from transformers import DebertaV2Tokenizer, DebertaV2ForTokenClassification
from data_process import DataTransform

dps = DataTransform()

test_df = pd.read_csv('./data/gt_test.csv', sep=',')

result = pd.DataFrame(columns=['processed_text', 'label'])

for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
    input_text = row['processed_text']

    entities, *_ = dps.get_entities(input_text)

    labels = dps.transform_text_labels(input_text, entities)

    result.loc[len(result)] = [input_text, labels]

result.to_csv('./data/submission.csv', index=False)
