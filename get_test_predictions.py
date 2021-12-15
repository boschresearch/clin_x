# Part 4 of our pipeline:
# The trained model is applied to the test data
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
from collections import namedtuple
from tqdm import tqdm

from flair.data import Sentence
from flair.models import SequenceTagger

from utils import *

Annotation = namedtuple('Annotation', ['tid', 'type', 'start', 'end', 'text'])
AnnToken = namedtuple('Token', ['idx', 'text', 'begin', 'end', 'label'])

parser = argparse.ArgumentParser()
parser.add_argument('--name', '-n', type=str, default='test', help='Experiment name used for logging')
parser.add_argument('--conll_path', type=str, required=True, help='Path to test files in conll format')
parser.add_argument('--out_path', type=str, required=True, help='Path to output files')
parser.add_argument('--tag_type', type=str, default='ner', help='Tag type to be used in flair models')
parser.add_argument('--mini_batch_size', type=int, default=16)

args = parser.parse_args()
args.conll_path = args.conll_path if args.conll_path.endswith('/') else args.conll_path + '/'
args.out_path = args.out_path if args.out_path.endswith('/') else args.out_path + '/'

# load data
tag_type = args.tag_type

cached_model = {'name': None}
def load_model(experiment):
    """Load the flair SequenceTagger model. 
    cached_model can be used if the same model has to be used several times"""
    if experiment == cached_model['name']:
        print('Use cached model')
        return cached_model['model']
    try:
        print(f'Load model: {experiment}/best-model.pt')
        model = SequenceTagger.load(f'{experiment}/best-model.pt')
    except: 
        print(f'Fallback to: {experiment}/final-model.pt')
        model = SequenceTagger.load(f'{experiment}/final-model.pt')
    cached_model['name'] = experiment
    cached_model['model'] = model
    return model

def _convert_to_flair_sents(tokens):
    """Takes a list of sentences and converts them to flair Sentence objects"""
    flair_sents = []
    for sent in tokens:
        words = [t[0] for t in sent]
        new_s = Sentence(' '.join(words), use_tokenizer=False)
        flair_sents.append(new_s)
    return flair_sents

def _convert_to_flat_tokens(tokens):
    """Takes a list of sentences and converts them to AnnToken objects"""
    flat_tokens = []
    for sent in tokens:
        for token in sent:
            new_t = AnnToken(token[0], token[1], int(token[2]), int(token[3]), token[4])
            flat_tokens.append(new_t)
    return flat_tokens

def write_bio_file(out_file, tokens, labels, pred_labels=None):
    """Writes the predictions to a file in bio format"""
    with open(out_file, 'w') as fout:
        if pred_labels is None:
            for sid, _ in enumerate(tokens):
                for token, label in zip(tokens[sid], labels[sid]):
                    idx = token.text
                    dec_text = token.get_tag('decoded_text').value
                    begin = token.get_tag('begin').value
                    end = token.get_tag('end').value
                    fout.write(f'{idx}\t{dec_text}\t{begin}\t{end}\t{label}\n')
                fout.write('\n')
        else:
            for sid, _ in enumerate(tokens):
                for token, label, pred in zip(tokens[sid], labels[sid], pred_labels[sid]):
                    idx = token.text
                    dec_text = token.get_tag('decoded_text').value
                    begin = token.get_tag('begin').value
                    end = token.get_tag('end').value
                    fout.write(f'{idx}\t{dec_text}\t{begin}\t{end}\t{pred}\n')
                fout.write('\n')

def annotate_documents(model, conll_path, out_path, mini_batch_size=64, tag_type='ner'):
    """Function to read and process the documents."""
    docs = sorted([conll_path + x for x in os.listdir(conll_path) if x.endswith('.bio')])
    print(f'Annotating {len(docs)} documents')
    
    for doc in tqdm(docs):
        sents = load_file_as_flair_corpus(doc.replace(conll_path, ''), conll_path)
        y_gold = [[token.get_tag(tag_type).value for token in sent] for sent in sents]
        model.predict(sents, mini_batch_size=mini_batch_size)
        y_pred = [[token.get_tag(tag_type).value for token in sent] for sent in sents]
        tokens = [[token for token in sent] for sent in sents]
        
        write_bio_file(out_path + doc.replace(conll_path, ''), tokens, y_gold, y_pred)
        
        
model = load_model(args.name)
os.makedirs(args.out_path, exist_ok=True)
annotate_documents(model, args.conll_path, args.out_path, mini_batch_size=args.mini_batch_size)