# Alternative training pipeline: 
# Train the transformer model with a classification layer on top
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
import random
import os

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from flair.data import Sentence, Token
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from utils import prepare_corpus

parser = argparse.ArgumentParser()
parser.add_argument('--name', '-n', type=str, required=True, 
                    help='Experiment name used for logging')
parser.add_argument('--language', '-l', type=str, required=True, 
                    help='Dataset (language) identifier')
parser.add_argument('--task', '-t', type=str, required=True, 
                    help='Dataset (task) identifier')
parser.add_argument('--data_path', '-p', type=str, required=True, help='Path to data files')
parser.add_argument('--train_files', type=str, default='train.bio', help='train file OR comma-seperated list of split files')
parser.add_argument('--dev_file', type=str, default=None, help='Development file')
parser.add_argument('--test_file', type=str, default='test.bio', help='Development file')
parser.add_argument('--use_training_splits', '-s', action="store_true")
parser.add_argument('--reduce_size', '-r', type=float, default=1.0, 
                    help='Factor to downsample dataset')
parser.add_argument('--storage_path', type=str, default='taggers/', 
                    help='Basepath to store model weights and logs')

# Model parameters
parser.add_argument('--model', '-m', type=str, required=True, help='Path to transformer model')
parser.add_argument('--lstm_hidden_size', type=int, default=256)
parser.add_argument('--att_hidden_size', type=int, default=10)
parser.add_argument('--word_hidden_size', type=int, default=25)
parser.add_argument('--shape_hidden_size', type=int, default=25)
parser.add_argument('--unit_normalization', action="store_true")
parser.add_argument('--use_crf', action="store_true")
parser.add_argument('--use_rnn', action="store_true")

# Training parameters
parser.add_argument('--learning_rate', type=float, default=5.0e-6)
parser.add_argument('--mini_batch_size', type=int, default=16)
parser.add_argument('--mini_batch_chunk_size', type=int, default=16)
parser.add_argument('--max_epochs', type=int, default=20)
parser.add_argument('--train_wth_dev', action="store_true")

args = parser.parse_args()

tag_type = 'ner'
language = 'en' if 'i2b2' in args.data_path else 'es'
lang = language

# Copy datasplits to a single temporary file
if args.use_training_splits:
    tmp_filename = '_tmp_' + str(random.random())[2:] + '.bio'
    print('Create temporary file: ' + args.data_path + tmp_filename)
    with open(args.data_path + tmp_filename, 'w') as fout:
        for fname in args.train_files.split(','):
            print('Add content from: ' + args.data_path + fname)
            with open(args.data_path + fname, 'r') as fin:
                content = fin.read()
                fout.write(content + '\n\n')
    train_file = tmp_filename
else:
    train_file = args.train_files

# Read the corpus as a flair object
if language == 'es':
    columns = {0: 'text', 1: 'begin', 2: 'end', 3: tag_type}
else:
    columns = {0: 'text', 1: tag_type}
corpus = ColumnCorpus(
    args.data_path, columns, 
    train_file=train_file,
    dev_file=args.dev_file,
    test_file=args.test_file,
    document_separator_token='<DOCSTART>'
)

if args.reduce_size < 1.0:
    corpus = corpus.reduce_to_size(args.reduce_size)
prepare_corpus(corpus, tag_type)

if args.use_training_splits:
    os.remove(args.data_path + tmp_filename)
    print('Delete temporary file')

tag_dictionary = corpus.make_tag_dictionary(tag_type)
tag_set = set()
for label in tag_dictionary.get_items():
    if label not in ['O', '<START>', '<STOP>', '<unk>']:
        tag_set.add(label[2:])
print(sorted(tag_set))

#####################
## Load embeddings ##
#####################

transformer = TransformerWordEmbeddings(
    args.model,
    layers = '-1',
    fine_tune=True,
    use_context=False,
    context_dropout=0.,
)
transformer.max_subtokens_sequence_length = 512
transformer.stride = 0

tagger = SequenceTagger(
    hidden_size=args.lstm_hidden_size,
    embeddings=transformer,
    tag_dictionary=tag_dictionary,
    tag_type=tag_type,
    use_crf=args.use_crf,
    use_rnn=args.use_rnn,
)

trainer = ModelTrainer(tagger, corpus, D=domain_c, optimizer=AdamW)

trainer.train(
    args.storage_path + args.name,
    learning_rate=args.learning_rate,
    max_epochs=args.max_epochs,
    train_with_dev=args.train_wth_dev,
    shuffle=True,
    mini_batch_size=args.mini_batch_size,
    mini_batch_chunk_size=args.mini_batch_chunk_size,
    scheduler=OneCycleLR,
    min_learning_rate=1e-8,
    monitor_test=True,
)

