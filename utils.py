# Different helper functions for data loading, prepatation and conversion
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

from flair.data import Sentence, Token
from flair.datasets import ColumnCorpus

def _cut_dataset(ds,  maxlen=300):
    def split_into_two(sent, k):
        s1, s2 = Sentence(), Sentence()
        s1._previous_sentence = sent._previous_sentence
        s1._next_sentence = s2
        s2._previous_sentence = s1
        s2._next_sentence = sent._next_sentence
        try:
            sent._previous_sentence._next_sentence = s1
        except:
            pass
        try:
            sent._next_sentence._previous_sentence = s2
        except:
            pass
        
        for tid, token in enumerate(sent.tokens):
            nt = Token(token.text)
            for tag_type in token.annotation_layers:
                nt.add_label(
                    tag_type,
                    token.get_tag(tag_type).value,
                    token.get_tag(tag_type).score,
                )

            if tid < k:
                s1.add_token(nt)
            else:
                s2.add_token(nt)
        return s1, s2

    tmp_sents = []
    for sent in ds:
        while len(sent) > maxlen:
            s1, sent = split_into_two(sent, maxlen)
            tmp_sents.append(s1)
        tmp_sents.append(sent)
    return tmp_sents

def prepare_corpus(corpus, tag_type):
    print('Cutting corpus to sentence length of 300 subwords')
    corpus._train = _cut_dataset(corpus._train)
    corpus._dev   = _cut_dataset(corpus._dev)
    corpus._test  = _cut_dataset(corpus._test)
    
    
def load_file_as_flair_corpus(train_file, data_path, tag_type='ner'):
    columns = {0: 'text', 1: "decoded_text", 2: 'begin', 3: 'end', 4: tag_type}
    corpus = ColumnCorpus(
        data_path, columns, 
        train_file=train_file,
        dev_file=train_file,
        test_file=train_file,
        tag_to_bioes=tag_type,
        document_separator_token='<DOCSTART>'
    )
    prepare_corpus(corpus, tag_type)
    return corpus._train


def _prepare_bio_elements(cur_elements, columns, keep, inflate, adjust_spaccc_merges):
    if adjust_spaccc_merges:
        assert 'token' in columns
        token_pos = columns.index('token') if keep == 'all' else keep.index('token')

        cur_elements_tmp = []
        for entry in cur_elements:
            token = entry[token_pos]
            if '_' in token:
                s = token.split('_')
                for subtoken in s:
                    e = [x for x in entry]
                    e[token_pos] = subtoken
                    cur_elements_tmp.append(e)
            else:
                cur_elements_tmp.append(entry)
        cur_elements = cur_elements_tmp

    if inflate and len(cur_elements[0]) == 1:
        cur_elements = [x[0] for x in cur_elements]
    return cur_elements

def read_bio_file(file_path, columns=['token', 'label'], keep='all', delimiter=None, inflate=False, adjust_spaccc_merges=False):
    lines = []
    with open(file_path, 'r') as fp:
        cur_elements = []
        
        for line in fp:
            line = line.strip()
            if line:
                if delimiter is None:
                    delimiter = '\t' if '\t' in line else ' '
                s = line.split(delimiter)
                if keep == 'all':
                    cur_elements.append(s)
                else:
                    s_tmp = []
                    for c, column in enumerate(columns):
                        if column in keep:
                            s_tmp.append(s[c])
                    cur_elements.append(s_tmp)
            elif len(cur_elements) > 0:
                cur_elements = _prepare_bio_elements(cur_elements, columns, keep, inflate, adjust_spaccc_merges)
                lines.append(cur_elements)
                cur_elements = []
        if len(cur_elements) > 0:
            cur_elements = _prepare_bio_elements(cur_elements, columns, keep, inflate, adjust_spaccc_merges)
            lines.append(cur_elements)
    return lines
    
    
def read_file_into_sentences(fname):
    with open(fname, 'r') as fin:
        content = fin.read().splitlines()
        all_ids, all_texts, all_labels = [[]], [[]], [[]]
        all_begins, all_ends = [[]], [[]]
        for line in content:
            if len(line.strip()) > 0:
                ids, text, begin, end, pred_label = line.split('\t')
                if pred_label == '<unk>':
                    print(fname)
                #if pred_label != 'O' and not label_filter in pred_label:
                #    print(f"converted {pred_label} to O")
                #    pred_label = 'O'
                
                all_ids[-1].append(ids)
                all_texts[-1].append(text)
                all_labels[-1].append(pred_label)
                all_begins[-1].append(begin)
                all_ends[-1].append(end)
            elif len(all_ids[-1]) > 0:
                all_ids.append([])
                all_texts.append([])
                all_labels.append([])
                all_begins.append([])
                all_ends.append([])
        return all_ids, all_texts, all_labels, all_begins, all_ends
