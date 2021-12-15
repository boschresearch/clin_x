# Part 1 of our training pipeline:
# Tokenize the training and testing files. 
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
import re

from tqdm import tqdm
from transformers import AutoTokenizer

from sentencesplit import sentencebreaks_to_newlines
NEWLINE_TERM_REGEX = re.compile(r'(.*?\n)')

from collections import namedtuple
Token = namedtuple('Token', 'encID text lemma pos doc_key sent_id token_id start end labels')
Annotation =namedtuple('Annotation', ['tid', 'type', 'start', 'end', 'text'])

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, default='xlm-roberta-large', help='Path to/Name of huggingface model')
parser.add_argument('--max_nested_level', type=int, default=1, help='Adds nested annotations up to level X')
parser.add_argument('--input_path', type=str)
parser.add_argument('--output_path', type=str)
args = parser.parse_args()


def read_file(fname):
    """Step 1: Read the BRAT standoff files. """
    with open(fname, 'r', encoding='utf-8') as fin:
        content = fin.read()
    #if content[0] == '﻿':
    #    return content[1:]
    return content


def _get_index_for_token(text, token, offset):
    """Get the text offset for a single token"""
    l = len(token)
    
    for s in range(offset, len(text)):
        if text[s:s+l] == token:
            return s, s+l
        
        if s > offset +100:
            raise ValueError('Takes too long')
            
        #
        # There are some special tokens converted by XLM-R that have to be reconverted
        #
        
        # simple replacements
        for org, rep, off in [('º', 'o', 0), ('ª', 'a', 0), ('´', '_', 0),
                              ('µ', 'μ', 0), ('ü', 'ü', 0), ('ñ', 'ñ', 0),
                              ('²', '2', 0), ('³', '3', 0), ('É', 'É', 0),
                              ('ö', 'ö', 0), ('Í', 'Í', 0), ('Ó', 'Ó', 0),
                              ('Ú', 'Ú', 0), ('´', '▁́', -1)]:
            if rep in token and org in text[s:s+l]:
                return s, s+l+off
        
        # fixes problems where XLM-R replacement is longer/shorter than real text
        if token == '1⁄2' and text[s:s+1] == '½':
            return s, s+1
        if token == '1⁄4' and text[s:s+1] == '¼':
            return s, s+1
        if token == '3⁄4' and text[s:s+1] == '¾':
            return s, s+1
        if token == '...' and text[s:s+1] == '…':
            return s, s+1
        if token == '...)' and text[s:s+2] == '…)':
            return s, s+2
        if token == '...).' and text[s:s+3] == '…).':
            return s, s+3
        if token == '"...' and text[s:s+2] == '"…':
            return s, s+2
        if token == '<unk>':
            return s, s+1
        if token == 'Á' and text[s:s+2] == 'Á':
            return s, s+2
        if token == 'ñ' and text[s:s+2] == 'ñ':
            return s, s+2
        if token == 'É' and text[s:s+2] == 'É':
            return s, s+2
        if token == 'RÁ' and text[s:s+3] == 'RÁ':
            return s, s+3
        if token == 'Í' and text[s:s+2] == 'Í':
            return s, s+2
        if token == 'Ó' and text[s:s+2] == 'Ó':
            return s, s+2
        if token == 'è' and text[s:s+2] == 'è':
            return s, s+2
        if token == 'Ú' and text[s:s+2] == 'Ú':
            return s, s+2
        if '™' in text[s-1:] and token == text.replace('™', 'TM')[s:s+l]:
            return s, s+l-1
        if 'ŀ' in text[s-1:] and token == text.replace('ŀ', 'l·')[s:s+l]:
            return s, s+l-1
        
    raise ValueError('Cannot match token: ' + token)
    
def get_offsets(text, tokens, offset):
    """Get the text offset for a list of tokens"""
    found = []
    try:
        for token in tokens:
            if token[0] == '▁': # Could be a BPE token
                try:
                    s1, e1 = _get_index_for_token(text, token, offset)
                except:
                    s1, e1 = len(text)+2, len(text)+3

                try: # remove BPE markup
                    s2, e2 = _get_index_for_token(text, token[1:], offset)
                except:
                    s2, e2 = len(text)+2, len(text)+3
                # Check which one is the real match
                
                if s1 <= s2:
                    assert s1 <= len(text)
                    found.append((s1, e1))
                    offset = e1
                else:
                    assert s2 <= len(text)
                    found.append((s2, e2))
                    offset = e2
                    
            else:
                s, e = _get_index_for_token(text, token, offset)
                assert s <= len(text)
                found.append((s, e))
                offset = e     
    except:
        raise ValueError('Tokenization problem')
    return found
    
def text_to_conll_columns(xlm_tokenizer, txt_content, doc_key='-'):
    """Step 2: Split text into sentences and tokens
    Convert plain text into CoNLL format."""
    sentences = []
    for l in txt_content.splitlines():
        l = sentencebreaks_to_newlines(l)
        sentences.extend([s for s in NEWLINE_TERM_REGEX.split(l) if s])
    
    document = [[Token('<DOCSTART>', '<DOCSTART>', '-x-', '-y-', doc_key, -1, -1, 0, 0, [])]]
    last_offset = 0
    for s_id, s in enumerate(sentences):
        enc = xlm_tokenizer.encode(s)
        org_tokens = xlm_tokenizer.convert_ids_to_tokens(enc)
        if len(org_tokens) == 0:
            continue
        tokens = org_tokens[1:-1] # strip special tokens
        if len(tokens) == 0:
            continue
        offsets = get_offsets(txt_content, tokens, last_offset)
        
        all_tokens = [(str(enc[t_id+1]), t, '-x-', '-y-', 
                   doc_key, s_id, t_id, 
                   offsets[t_id][0], offsets[t_id][1]) 
                  for t_id, t in enumerate(tokens)]
        
        # token now maps to <Text, POS, SentID, TokenID, OffsetStart, OffsetEnd>
        tokens = [Token(*s, []) for s in all_tokens]
        document.append(tokens)
        last_offset = tokens[-1].end
    return document


ANNOTATION_REGEX = re.compile(r'(T\d+)\s+(\S*?)\s(\d+)\s(\d+)\s(.*)', re.MULTILINE)

def read_annotations(ann_content):
    """Step 3: Read annotations and check for nested mentions.
    Read annotations from a brat standoff annotation file. """
    annotations = []
    for m in ANNOTATION_REGEX.finditer(ann_content):
        t_id = m.group(1)
        entity_type = m.group(2)
        ann_begin = int(m.group(3))
        ann_end = int(m.group(4))
        entity_text = m.group(5)
        annotations.append(Annotation(t_id, entity_type, ann_begin, ann_end, entity_text))
    return annotations

def seperate_nested_entities(annotations, max_nested_level=0):
    """Create nested hierarchy"""
    nested_annotations = {}
    nested_level = 0

    while len(annotations) > 0 and nested_level < max_nested_level:
        nested_annotations[nested_level] = []
        eliminate = {}
    
        for a1 in annotations:
            for a2 in annotations:
                if a1 is a2:
                    continue
                if a2.start >= a1.end or a2.end <= a1.start:
                    continue
                # eliminate shorter
                if a1.end - a1.start > a2.end - a2.start:
                    eliminate[a2] = True
                else:
                    eliminate[a1] = True
                
        nested_annotations[nested_level] = [a for a in annotations if a not in eliminate]
        annotations = [a for a in annotations if a in eliminate] 
        nested_level += 1
        
    nested_level = len(nested_annotations)
    if nested_level < max_nested_level:
        for n in range(nested_level, max_nested_level+1):
            nested_annotations[n] = []
    return nested_annotations, nested_level


def attach_annotations_to_text(document, annotations):
    """Step 4: Attach annotations to document"""
    labels = ['O' for _ in range(document[-1][-1].end)]
    for a in annotations:
        #print(a)
        labels[a.start] = 'B-' + a.type
        for i in range(a.start+1, a.end):
            labels[i] = 'I-' + a.type
       
    prev_annotation = 't-1'
    for sid, sentence in enumerate(document):
        if sid == 0: # Docstart
            for token in sentence:
                assert token.text == '<DOCSTART>'
                token.labels.append('O')
        else:
            for token in sentence:
                token.labels.append(labels[token.start])
            
def get_conll_format(document, print_output=False):
    """Write tokens into real conll format.
    Each line consists of the follwing columns:
    0: Text, 
    1: Lemma, 
    2: POS-tag, 
    3: DocKey, 
    4: #sent, 
    5: #token,
    6: offset_begin, 
    7: offset_end,  
    8: ner_label,
    9: nested_ner_label_1 if available
    ...
    """
    conll_content = []
    for sentence in document:
        for token in sentence:
            labels = '\t'.join(token.labels)
            fields = [str(t) for t in token]
            columns = '\t'.join([fields[0], fields[0+1], fields[6+1], fields[7+1]])
            conll_content.append(f'{columns}\t{labels}')
        conll_content.append('')
    if print_output:
        for line in conll_content:
            print(line)
    return '\n'.join(conll_content)


def check_annotations(document, nested_annotations, nested_level=1):
    """Step 5: Check annotations for boundary problems"""
    all_annotations = []
    for n in range(nested_level):
        all_annotations.extend([a for a in nested_annotations[n]])
        
    problems = 0
    for a in all_annotations:
        tokens_to_consider = []
        found_start, found_end = False, False
    
        for sent in document:
            for t in sent:
                if t.start == a.start:
                    found_start = True
                if t.end == a.end:
                    found_end = True
            
                # check if this token is helpful for debugging
                if abs(t.start - a.start) < 10 or abs(t.end - a.end) < 10:
                    tokens_to_consider.append(t)
                
            if found_start and found_end:
                break
            
        if not found_start or not found_end:
            problems += 1
            print(f'Problem for Annotation (start: {found_start}; end: {found_end}) in file: {t.doc_key}')
            print(a)
            for t in tokens_to_consider:
                print(t)
            raise ValueError()
    return -problems


def tokenize_document(txt_file, ann_file, xlm_tokenizer, doc_key='-'):
    """All in one-function"""
    txt_content = read_file(txt_file)
    try:
        ann_content = read_file(ann_file)
    except:
        print('Not annotation file: ' + ann_file)
        ann_content = ''
    
    document = text_to_conll_columns(xlm_tokenizer, txt_content, doc_key)
    annotations = read_annotations(ann_content)
    nested_annotations, nested_level = seperate_nested_entities(annotations, args.max_nested_level)
    for n in range(args.max_nested_level):
        attach_annotations_to_text(document, nested_annotations[n])
    conll_file = get_conll_format(document)
    #check_annotations(document, nested_annotations, nested_level)
    
    return conll_file


def process_files(ann_path, out_path, hf_tokenizer, txt_path=None):
    """Read standoff files from directory and converts them to conll files"""
    if txt_path is None:
        txt_path = ann_path
    if out_path is not None:
        os.makedirs(out_path, exist_ok=True)
        
    ann_files = sorted([ann_path + x for x in os.listdir(ann_path) if x.endswith('.ann')])
    txt_files = sorted([txt_path + x for x in os.listdir(txt_path) if x.endswith('.txt')])
    
    if len(ann_files) == 0:
        ann_files = [fname.replace('.txt', '.ann') for fname in txt_files]
        print('There are no annotation files')
        
    t = tqdm(total=len(ann_files))
    num_proc = 0
    for ann_file, txt_file in zip(ann_files, txt_files):
        doc_key = txt_file.split('/')[-1][:-4]
        #print(doc_key)
        conll_format = tokenize_document(txt_file, ann_file, hf_tokenizer, doc_key)

        with open(out_path + doc_key + '.bio', 'w') as fout:
            fout.write(conll_format)
                
        num_proc += 1
        t.update()


    
    
if __name__ == "__main__":    
    print('Load transformer model from: ' + args.model)
    hf_tokenizer = AutoTokenizer.from_pretrained(args.model)
    process_files(args.input_path, args.output_path, hf_tokenizer)
    