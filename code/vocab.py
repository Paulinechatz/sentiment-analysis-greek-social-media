import transformers
from transformers import RobertaTokenizerFast
from tokenizers import ByteLevelBPETokenizer
import torch
import argparse as ap
import pandas as pd
import os

parser = ap.ArgumentParser(description='Train a BPE tokenizer')

parser.add_argument('-i', '--input', type=str, help='Input corpus (in json format)')
parser.add_argument('-v', '--vocab', type=str, help='Vocabulary File directory')
parser.add_argument('-o', '--output', type=str, help='Tokenizer output directory')
parser.add_argument('--size', type=int, help='Vocabulary size', default=30_522)
parser.add_argument('--min_freq', type=int, help='Minimum Frequency', default=2)
parser.add_argument('--text', type=str, help='Text attribute in json', default='text')

args = parser.parse_args()

df = pd.read_json(args.input)

# Build a tokenizer from scratch

tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train_from_iterator(iterator=df[args.text].to_list(), vocab_size=args.size, min_frequency=args.min_freq,
                              special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]) 

if not os.path.exists(args.vocab): # -p1: punct / -p2: nopunct / p3: nopunct+upper / p4: punct+upper
    os.mkdir(args.vocab)

tokenizer.save_model(args.vocab)

tokenizer = RobertaTokenizerFast.from_pretrained(args.vocab, max_len=512)

if not os.path.exists(args.output):
    os.mkdir(args.output)

tokenizer.save_pretrained(args.output)