import torch
from pathlib import Path
from tqdm.auto import tqdm
import argparse as ap
from transformers import RobertaTokenizerFast, DataCollatorForLanguageModeling, RobertaConfig, RobertaForMaskedLM, TrainingArguments, Trainer
import os

parser = ap.ArgumentParser(description='MLM from scratch')

parser.add_argument('-v', '--vocab', type=str, help='Vocabulary file')
parser.add_argument('-ct', '--corpus_train', type=str, help='Output corpus.txt directory')
parser.add_argument('-ce', '--corpus_eval', type=str, help='Output corpus.txt directory')
parser.add_argument('-chkpt', '--checkpoint', type=str, help='Resume training from Checkpoint (path)')
parser.add_argument('-o', '--output', type=str, help='Output directory')
parser.add_argument('--text', type=str, help='Text attribute in json', default='text')

parser.add_argument('-td', '--train_dir', type=str, help='Training directory', default='./results')
parser.add_argument('-ld', '--log_dir', type=str, help='Log directory', default='./logs')

parser.add_argument('--epochs', type=int, help='Number of epochs', default=1)
parser.add_argument('--batch_size', type=int, help='Batch size', default=10)
parser.add_argument('--steps', type=int, help='Save steps', default=10_000)
parser.add_argument('--warmup', type=int, help='Warmup steps', default=500)
parser.add_argument('--wd', type=float, help='Weight decay', default=0.01)

args = parser.parse_args()

with open(args.corpus_train, 'r', encoding='utf-8') as fp:
    train_lines = fp.read().split('\n')

with open(args.corpus_eval, 'r', encoding='utf-8') as fp:
    eval_lines = fp.read().split('\n')

# initialize tokenizer using the tokenizer we initialized and saved to file

tokenizer = RobertaTokenizerFast.from_pretrained(args.vocab, max_length=512)

# initialize data collector

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

train_samples = tokenizer(train_lines, max_length=512, truncation=True,  add_special_tokens=True) # no padding, DataCollator performs dynamic padding for each batch
eval_samples = tokenizer(eval_lines, max_length=512, truncation=True,  add_special_tokens=True)

train_encodings = []
eval_encodings = []

for s in train_samples['input_ids']: # create list of dictionaries
  encoding = {'input_ids': s}   
  train_encodings.append(encoding)

for s in eval_samples['input_ids']: # create list of dictionaries
  encoding = {'input_ids': s}   
  eval_encodings.append(encoding)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings

    def __len__(self):
        # return the number of samples
        return len(self.encodings)

    def __getitem__(self, idx):
        # return dictionary of input_ids for index idx
        return {key: tensor for key, tensor in self.encodings[idx].items()}

train_dataset = Dataset(train_encodings)
eval_dataset = Dataset(eval_encodings)

# building a model: create a RoBERTa config object

config = RobertaConfig(
    vocab_size=30_522,  # same as tokenizer vocab set in vocab.ipynb
    max_position_embeddings=514, # max_length=512 + <s>, </s> tokens
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1
)

model = RobertaForMaskedLM.from_pretrained(pretrained_model_name_or_path=args.checkpoint) #  initialize a RoBERTa model with a language modeling head 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device) # move model over to the selected device

if not os.path.exists(args.train_dir): # make train/log directories
    os.mkdir(args.train_dir)

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

training_args = TrainingArguments(
    report_to="none",
    output_dir=args.train_dir,          # output directory
    num_train_epochs=args.epochs,              # total number of training epochs
    group_by_length=True,                      # smart batching
    per_device_train_batch_size=args.batch_size,  # batch size per device during training
    per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
    warmup_steps=args.warmup,                # number of warmup steps for learning rate scheduler
    weight_decay=args.wd,               # strength of weight decay
    logging_dir=args.log_dir,            # directory for storing logs
    logging_strategy='epoch',
    evaluation_strategy= 'epoch',
    save_strategy='epoch'
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train(resume_from_checkpoint=args.checkpoint)
trainer.save_model(args.output)