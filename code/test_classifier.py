from sklearn.metrics import classification_report
from transformers import Trainer, AutoTokenizer, AutoModelForSequenceClassification
import argparse as ap
import pandas as pd
import torch
import os

parser = ap.ArgumentParser(description='Test a sentiment classifier')

parser.add_argument('-v', '--vocab', type=str, help='Vocabulary')
parser.add_argument('-c', '--classifier', type=str, help='Classifier')
parser.add_argument('-ts', '--test-dataset', type=str, help="Palo test dataset (in csv format)")
parser.add_argument('--text', type=str, help='text field in the datasets', default='text')
parser.add_argument('--sentiment', type=str, help='sentiment field in the datasets', default='sentiment')

args = parser.parse_args()

df_test = pd.read_csv(args.test_dataset)

tokenizer =  AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.vocab, max_len=512)

test_data = tokenizer(df_test[args.text].to_list(), max_length=512, truncation=True,
                   padding='max_length', add_special_tokens=True, return_tensors='pt')

test_labels = df_test[args.sentiment].to_list()

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings

    def __len__(self):
        # return the number of samples
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {key: tensor[i] for key, tensor in self.encodings.items()}

test_encodings = {'input_ids': test_data['input_ids'], 'attention_mask': test_data['attention_mask'], 'labels': torch.tensor(test_labels)}
test_dataset = Dataset(test_encodings)

model = AutoModelForSequenceClassification.from_pretrained(args.classifier)
trainer = Trainer(model=model)

results = trainer.predict(test_dataset)
preds = [i.index(max(i)) for i in results.predictions.tolist()]

dirname=os.path.dirname(f'{args.classifier}')
basename=os.path.basename(dirname)

folder=f'{dirname}/{basename}'

if not os.path.exists(folder):
  os.mkdir(folder)

real_values = f'{folder}/real_values.txt'
predictions = f'{folder}/pred.txt'

with open(real_values, 'w') as f:
  for line in test_labels:
      f.write(f"{line}\n")

with open(predictions, 'w') as f:
  for line in preds:
    f.write(f"{line}\n")

print(classification_report(test_labels, preds, digits=4))