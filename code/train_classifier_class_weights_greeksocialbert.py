import torch
import pandas as pd
import argparse as ap
from torch import nn
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

parser = ap.ArgumentParser(description='Train a sentiment classifier')

parser.add_argument('-tr', '--train_dataset', type=str, help="Palo train dataset (in csv format)")
parser.add_argument('-val', '--val_dataset', type=str, help="Palo validation dataset (in csv format)")
parser.add_argument('-v', '--vocab', type=str, help='Vocabulary File')
parser.add_argument('-lm', '--language_model', type=str, help='Language model (name or path)')
parser.add_argument('-o', '--output', type=str, help='Classifier output directory')

parser.add_argument('--text', type=str, help='text field in the datasets', default='text')
parser.add_argument('--sentiment', type=str, help='sentiment field in the datasets', default='sentiment')

parser.add_argument('-td', '--train_dir', type=str, help='Training directory', default='./results')
parser.add_argument('-ld', '--log_dir', type=str, help='Log directory', default='./logs')

parser.add_argument('--epochs', type=int, help='Number of epochs', default=5)
parser.add_argument('--lr', type=float, help='Learning rate', default=5e-5)
parser.add_argument('--batch_size', type=int, help='Batch size', default=16)
parser.add_argument('--warmup', type=int, help='Warmup steps', default=500)
parser.add_argument('--wd', type=float, help='Weight decay', default=0.01)

args = parser.parse_args()

df_train = pd.read_csv(args.train_dataset)
df_val = pd.read_csv(args.val_dataset)

# initialize trained tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.vocab, max_len=512)

# tokenize
train_data = tokenizer(df_train[args.text].to_list(), max_length=512, truncation=True,
                   padding='max_length', add_special_tokens=True, return_tensors='pt')

val_data = tokenizer(df_val[args.text].to_list(), max_length=512, truncation=True,
                   padding='max_length', add_special_tokens=True, return_tensors='pt')

labels_train = df_train[args.sentiment].to_list()
labels_val = df_val[args.sentiment].to_list()

encodings_train = {'input_ids': train_data['input_ids'], 'attention_mask': train_data['attention_mask'], 'labels': torch.tensor(labels_train)}
encodings_val = {'input_ids': val_data['input_ids'], 'attention_mask': val_data['attention_mask'], 'labels': torch.tensor(labels_val)}

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

train_dataset=Dataset(encodings_train)
val_dataset=Dataset(encodings_val)

# sequence classification parameters
id2label = {0: 'neutral', 1: 'positive', 2: 'negative'}
label2id = {'neutral': 0, 'positive': 1, 'negative': 2}

#set class_weights for imbalanced dataset
class_weights = (1-(df_train[args.sentiment].value_counts(normalize=True).sort_index())).values
class_weights = torch.from_numpy(class_weights).float().to("cuda")

# define accuracy metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, labels=[1,2], average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

if not os.path.exists(args.train_dir):
    os.mkdir(args.train_dir)

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

class WeightedLossTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False):
    # feed inputs to model
    outputs = model(**inputs)
    # extract logits
    logits = outputs.get("logits")
    # extract labels
    labels = inputs.get("labels")
    # define loss func with class weights of imbalanced dataset
    loss_func = nn.CrossEntropyLoss(weight=class_weights)
    # compute loss
    loss = loss_func(logits, labels)
    return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir=args.train_dir,          # output directory
    num_train_epochs=args.epochs,              # total number of training epochs
    learning_rate=args.lr,                      #learning rate
    per_device_train_batch_size=args.batch_size,  # batch size per device during training
    per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
    warmup_steps=args.warmup,                # number of warmup steps for learning rate scheduler
    weight_decay=args.wd,               # strength of weight decay
    logging_dir=args.log_dir,            # directory for storing logs
    evaluation_strategy='epoch',
    logging_strategy='epoch',
    save_strategy='epoch',
    report_to = "none",
    #load_best_model_at_end=True
)

model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model, num_labels = len(id2label), 
                                                         id2label=id2label, label2id=label2id)

# freeze pretrained model weights
for param in model.bert.parameters(): 
  param.requires_grad = False

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# move model over to the selected device
model.to(device)

trainer = WeightedLossTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    compute_metrics=compute_metrics,     # model evaluation metrics
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()
trainer.save_model(args.output)
