# Part 3 of our training pipeline:
# The actual training of the NER model on subword-level with context information
# Copyright (c) 2021 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import os
import random

import torch
from torch.optim.lr_scheduler import OneCycleLR

from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from utils import prepare_corpus

from embeddings import SubwordTransformerWordEmbeddings

parser = argparse.ArgumentParser()
parser.add_argument('--name', '-n', type=str, default='test', help='Experiment name used for logging')
parser.add_argument('--data_path', '-p', type=str, required=True, help='Path to data files')
parser.add_argument('--train_files', '-t', type=str, required=True, help='Comma-seperated list of training files')
parser.add_argument('--dev_file', '-d', type=str, required=True, help='Development file')
parser.add_argument('--model', '-m', type=str, required=True, help='Path to transformer model')
parser.add_argument('--context', '-c', type=int, default=100, help='Context size of transformer model')
parser.add_argument('--storage_path', type=str, default='taggers/', help='Basepath to store model weights and logs')
parser.add_argument('--tag_type', type=str, default='ner', help='Tag type to be used in flair models')

# Model parameters
parser.add_argument('--use_crf', action="store_true")
parser.add_argument('--use_rnn', action="store_true")
parser.add_argument('--hidden_size', type=int, default=256, help='The RNN hidden size if --use_rnn is active')
parser.add_argument('--no_finetuning', action="store_true")

# Training parameters
parser.add_argument('--learning_rate', type=float, default=2.0e-5)
parser.add_argument('--mini_batch_size', type=int, default=16)
parser.add_argument('--mini_batch_chunk_size', type=int, default=16)
parser.add_argument('--max_epochs', type=int, default=20)
parser.add_argument('--train_with_dev', action="store_true")

args = parser.parse_args()
args.data_path = args.data_path if args.data_path.endswith('/') else args.data_path + '/'
args.storage_path = args.storage_path if args.storage_path.endswith('/') else args.storage_path + '/'

# load data
tag_type = args.tag_type

# Copy datasplits to a single temporary file
tmp_filename = '_tmp_' + str(random.random())[2:] + '.bio'
print('Create temporary file: ' + args.data_path + tmp_filename)
with open(args.data_path + tmp_filename, 'w') as fout:
    for fname in args.train_files.split(','):
        print('Add content from: ' + args.data_path + fname)
        with open(args.data_path + fname, 'r') as fin:
            content = fin.read()
            fout.write(content + '\n\n')

# Read the corpus as a flair object
columns = {0: 'text', 1: "decoded_text", 2: 'begin', 3: 'end', 4: tag_type}
corpus = ColumnCorpus(
    args.data_path, columns, 
    train_file=tmp_filename,
    dev_file=args.dev_file,
    test_file=args.dev_file,
    tag_to_bioes=tag_type,
    document_separator_token='<DOCSTART>'
)
prepare_corpus(corpus, tag_type)

os.remove(args.data_path + tmp_filename)
print('Delete temporary file')

tag_dictionary = corpus.make_tag_dictionary(tag_type)
tag_set = set()
for label in tag_dictionary.get_items():
    if label not in ['O', '<START>', '<STOP>', '<unk>']:
        tag_set.add(label[2:])
print(sorted(tag_set))


transformer_model = args.model
print('Use transformer: ' + transformer_model)

embeddings = SubwordTransformerWordEmbeddings(
    model=transformer_model,
    layers="-1",
    subtoken_pooling="first",
    fine_tune=not args.no_finetuning,
    use_context=args.context,
)

tagger: SequenceTagger = SequenceTagger(
    hidden_size=args.hidden_size,
    embeddings=embeddings,
    tag_dictionary=tag_dictionary,
    tag_type=tag_type,
    use_crf=args.use_crf,
    use_rnn=args.use_rnn,
    reproject_embeddings=False,
)

trainer = ModelTrainer(tagger, corpus, optimizer=torch.optim.AdamW)

trainer.train(
    args.storage_path + args.name,
    learning_rate=args.learning_rate,
    mini_batch_size=args.mini_batch_size,
    mini_batch_chunk_size=args.mini_batch_chunk_size,
    max_epochs=args.max_epochs,
    scheduler=OneCycleLR,
    embeddings_storage_mode='none',
    weight_decay=0.,
    monitor_test=True,
    train_with_dev=args.train_with_dev,
)
