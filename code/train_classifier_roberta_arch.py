import torch
import transformers
import pandas as pd
import argparse as ap
from torch import nn
import os
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel

parser = ap.ArgumentParser(description='Train a sentiment classifier - New Architectures')

parser.add_argument('-tr', '--train_dataset', type=str, help="Palo train dataset (in csv format)")
parser.add_argument('-val', '--val_dataset', type=str, help="Palo validation dataset (in csv format)")
parser.add_argument('-v', '--vocab', type=str, help='Vocabulary File')
parser.add_argument('-lm', '--language_model', type=str, help='Language model (name or path)')
parser.add_argument('-o', '--output', type=str, help='Classifier output directory', default='./results/text-classification')

parser.add_argument('--text', type=str, help='text field in the datasets', default='text')
parser.add_argument('--sentiment', type=str, help='sentiment field in the datasets', default='sentiment')

parser.add_argument('--arch', type=str, help='Model Architecture', default='1')
parser.add_argument('--class_weights', type=int, help='Class weights for imbalanced dataset', default=1)
parser.add_argument('--smax', type=int, help='Softmax activation', default=0)
parser.add_argument('--ln', type=int, help='Layer Nomralization', default=1)

parser.add_argument('--dropout', type=float, help='Layer dropout rate', default=0.1)
parser.add_argument('--dim', type=int, help='Hidden Layer dimension', default=768)

parser.add_argument('--epochs', type=int, help='Number of epochs', default=5)
parser.add_argument('--lr', type=float, help='Learning rate', default=0.001)
parser.add_argument('--batch_size', type=int, help='Batch size', default=16)

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

#set class_weights for imbalanced dataset
class_weights = (1-(df_train[args.sentiment].value_counts(normalize=True).sort_index())).values
class_weights = torch.from_numpy(class_weights).float().to("cuda")

#Architecture 1: second hidden layer with relu activation function + (optinal) layer nomarlization + softmax:

class SentimentClassifier_v1(nn.Module):

  def __init__(self, language_model, n_classes):
    super(SentimentClassifier_v1, self).__init__()

    self.roberta = AutoModel.from_pretrained(pretrained_model_name_or_path=args.language_model)
    self.fc = torch.nn.Linear(768, args.dim)
    self.classifier = torch.nn.Linear(args.dim, n_classes)
    self.layernorm = torch.nn.LayerNorm(args.dim)
    self.drop = torch.nn.Dropout(args.dropout)
    self.relu = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax(dim=1)
  
  def forward(self, input_ids, attention_mask):
    _, output = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
    output = self.drop(output)
    output = self.fc(output)
    if args.ln == 1: #flag
      output = self.layernorm(output)
    output = self.relu(output)
    output = self.drop(output)
    output = self.classifier(output)
    if args.smax==1: 
      output = self.softmax(output)
    return output

#Architecture 2: pooled output of last hidden state: ReLu activation + LayerNorm (optional):

class SentimentClassifier_v2(torch.nn.Module):
    def __init__(self, language_model, n_classes):
        super(SentimentClassifier_v2, self).__init__()
        self.roberta = AutoModel.from_pretrained(pretrained_model_name_or_path=args.language_model)
        self.fc = torch.nn.Linear(768, args.dim)
        self.layernorm = torch.nn.LayerNorm(args.dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(args.dropout)
        self.classifier = torch.nn.Linear(args.dim, n_classes)

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = roberta_output[0]
        pooler = hidden_state[:, 0]
        pooler = self.fc(pooler)
        if args.ln==1:
         pooler = self.layernorm(pooler)
        pooler = self.relu(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
    
#Architecture 3: Second hidden layer with relu activation function + (optional) layer nomarlization:

class SentimentClassifier_v3(nn.Module):

  def __init__(self, language_model, n_classes):
    super(SentimentClassifier_v3, self).__init__()

    self.roberta = AutoModel.from_pretrained(pretrained_model_name_or_path=args.language_model)
    self.fc1 = torch.nn.Linear(768, 768)
    self.fc2 = torch.nn.Linear(768, args.dim)
    self.classifier = torch.nn.Linear(args.dim, n_classes)
    self.layernorm1 = torch.nn.LayerNorm(768)
    self.layernorm2 = torch.nn.LayerNorm(args.dim)
    self.drop = torch.nn.Dropout(args.dropout)
    self.relu = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax(dim=1)
  
  def forward(self, input_ids, attention_mask):
    _, output = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
    output = self.drop(output)
    output = self.fc1(output)
    if args.ln==1:
     output = self.layernorm1(output)
    output = self.relu(output)
    output = self.drop(output)
    output = self.fc2(output)
    if args.ln==1:
     output = self.layernorm2(output)
    output = self.relu(output)
    output = self.drop(output)
    output = self.classifier(output)
    return output

#Architecture 4: Pooled output of last hidden state: Second hidden layern + ReLu activation + LayerNorm (optional):

class SentimentClassifier_v4(torch.nn.Module):
    def __init__(self, language_model, n_classes):
        super(SentimentClassifier_v4, self).__init__()
        self.roberta = AutoModel.from_pretrained(pretrained_model_name_or_path=args.language_model)
        self.fc1 = torch.nn.Linear(768, 768)
        self.fc2 = torch.nn.Linear(768, args.dim)
        self.layernorm1 = torch.nn.LayerNorm(768)
        self.layernorm2 = torch.nn.LayerNorm(args.dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(args.dropout)
        self.classifier = torch.nn.Linear(args.dim, n_classes)

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = roberta_output[0]
        pooler = hidden_state[:, 0]
        pooler = self.fc1(pooler)
        if args.ln==1:
         pooler = self.layernorm1(pooler)
        pooler = self.relu(pooler)
        pooler = self.dropout(pooler)
        pooler = self.fc2(pooler)
        if args.ln==1:
         pooler = self.layernorm2(pooler)
        pooler = self.relu(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
    
class Switch():
  def switch(self, model):
    default = "Invalid Model Architecture!"
    return getattr(self, 'arch_' + str(model), lambda:default)()
   
  def arch_1(self):
    model = SentimentClassifier_v1(args.language_model, 3)
    return model
  def arch_2(self):
    model = SentimentClassifier_v2(args.language_model, 3)
    return model
  def arch_3(self):
    model = SentimentClassifier_v3(args.language_model, 3)
    return model
  def arch_4(self):
    model = SentimentClassifier_v4(args.language_model, 3)
    return model
  
model_switch = Switch()
model=model_switch.switch(args.arch)

# freeze pretrained model weights
for param in model.roberta.parameters(): 
  param.requires_grad = False

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# move model over to the selected device
model.to(device)

EPOCHS=args.epochs
# activate training mode
model.train()
# initialize optimizer
optim = AdamW(model.parameters(), lr=args.lr) #default HF optimizer
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

total_steps = len(dataloader) * EPOCHS

scheduler = get_linear_schedule_with_warmup( #default HF scheduler
  optim,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

if args.class_weights == 1:
  loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device)
else:
  loss_fn = nn.CrossEntropyLoss().to(device)

def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler, 
  n_examples
):
  model = model.train()

  losses = []
  correct_predictions = 0
  
  loop = tqdm(data_loader,leave=True)

  for d in loop:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["labels"].to(device)

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)

    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    loop = tqdm(data_loader,leave=True)
    for d in loop:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["labels"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)

#--------training loop------------------

from collections import defaultdict

history = defaultdict(list)
if not os.path.exists(args.output):
    os.mkdir(args.output)

lm = os.path.basename(os.path.dirname(f'{args.language_model}'))

for epoch in range(EPOCHS):

  print(f'Epoch {epoch+1}/{EPOCHS}')
  print('-'*10)

  train_acc, train_loss = train_epoch(
    model,
    dataloader,    
    loss_fn, 
    optim, 
    device, 
    scheduler, 
    len(df_train)
  )

  print(f'Train loss {train_loss} accuracy {train_acc}')

  val_acc, val_loss = eval_model(
    model,
    val_dataloader,
    loss_fn, 
    device, 
    len(df_val)
  )

  print(f'Val loss {val_loss} accuracy {val_acc}')

  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)

  checkpoint = f'{args.output}/checkpoint-{epoch+1}' #create directory
  torch.save(model.state_dict(), checkpoint+'.bin')

folder=os.path.basename(f'{args.output}')
performance = f'{args.output}/{folder}'

if not os.path.exists(performance):
  os.mkdir(performance)

train_loss = f'{performance}/train_loss.txt'
val_loss = f'{performance}/val_loss.txt'
train_acc = f'{performance}/train_acc.txt'
val_acc = f'{performance}/val_acc.txt'

with open(train_loss, 'w') as f:
  for line in history['train_loss']:
      f.write(f"{line}\n")

with open(val_loss, 'w') as f:
  for line in history['val_loss']:
    f.write(f"{line}\n")

with open(train_acc, 'w') as f:
  for line in history['train_acc']:
      f.write(f"{line}\n")

with open(val_acc, 'w') as f:
  for line in history['val_acc']:
      f.write(f"{line}\n")
