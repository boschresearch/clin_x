# Part 5 of our pipeline:
# Different model predictions are combined in one ensemble model
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

import os
import sys
from utils import read_file_into_sentences

res_files = {}
out_path = sys.argv[1]
out_path = out_path if out_path.endswith('/') else out_path + '/'
os.makedirs(out_path, exist_ok=True)
print('Print ensemble predictions to: ' + out_path)
for model in sys.argv[2:]:
    model = model if model.endswith('/') else model + '/'
    print('Adding predictions from model: ' + model)
    res_files[model] = [f for f in os.listdir(model)]


predictions = {}
infos = {}
for model in res_files:
    for fname in res_files[model]:
        ids, texts, labels, begins, ends = read_file_into_sentences(model + fname)
        if fname not in predictions:
            predictions[fname] = []
            infos[fname] = (ids, texts, begins, ends)
        predictions[fname].append(labels)
        
        
from collections import Counter
import math

def resolve_annotation(counter, neutral='O', confidence=0):
    """Performs majority voting"""
    counter_sum = sum([c for i,c in counter.most_common()])
    if counter[neutral]-confidence >= int(math.ceil(counter_sum/2)):
        # Majority votes for no annotation
        return neutral
    else:
        max_count, max_label = 0, None
        for label, count in counter.most_common():
            if count > max_count and label != neutral:
                max_label = label
                max_count = count
        return max_label
    
def correct_annotation(labels):
    """Corrects mistakes in the BIOSE annotation schema"""
    corrected_labels = []
    
    if len(labels) == 1:
        corrected_labels.append('S-' + labels[0][2:])
    elif len(labels) > 1:
        type_counter = Counter()
        type_counter[labels[0][2:]] += 1
        for i in range(1, len(labels)-1):
            type_counter[labels[i][2:]] += 1
        type_counter[labels[-1][2:]] += 1
        
        if len(type_counter) == 1:
            use_type = type_counter.most_common()[0][0]
        else:
            opt_type_a, opt_count_a = type_counter.most_common()[0]
            opt_type_b, opt_count_b = type_counter.most_common()[1]
            
            if opt_count_a > opt_count_b:
                use_type = opt_type_a
            else:
                assert opt_count_a == opt_count_b
                if labels[0][2:] == opt_type_b:
                    use_type = opt_type_b
                else:
                    use_type = opt_type_a
                    
        corrected_labels.append('B-' + use_type)
        for i in range(1, len(labels)-1):
            corrected_labels.append('I-' + use_type)
        corrected_labels.append('E-' + use_type)
        
    corrected_labels = [l.replace('E-', 'I-').replace('S-', 'B-') for l in corrected_labels]
        
    if labels != corrected_labels:
        print(f'corrected: {labels} -> {corrected_labels}')
    else:
        pass
    assert len(corrected_labels) == len(labels)
    return corrected_labels
    
    
def correct_tag_sequence(tag_sequence): 
    """Corrects mistakes in the BIOSE annotation schema"""
    ext_tag_sequence = ['O'] + tag_sequence + ['O']
    corrected_sequence = []

    annotation_labels = []
    for i in range(1, len(tag_sequence)+1):
        prev, curr, succ = ext_tag_sequence[i-1:i+2]

        starts_annotation = curr[0] in ['B']
        mid_annotation    = curr[0] in ['I']
        
        if starts_annotation and len(annotation_labels) > 0:
            corrected_sequence.extend(correct_annotation(annotation_labels))
            annotation_labels = []
        elif curr[0] == 'O' and len(annotation_labels) > 0:
            corrected_sequence.extend(correct_annotation(annotation_labels))
            annotation_labels = []
            
        if curr[0] == 'O':
            corrected_sequence.append(curr)
        elif starts_annotation:
            annotation_labels = [curr]
        elif mid_annotation:
            annotation_labels.append(curr)
            
    if len(annotation_labels) > 0:
        corrected_sequence.extend(correct_annotation(annotation_labels))
            
    assert len(tag_sequence) == len(corrected_sequence), (len(tag_sequence), len(corrected_sequence))
    return corrected_sequence
            
def check_tags(tag_sequence):
    """Can be used the validate the tag sequence. 
    Is skipped in our setting. """
    return True
    ext_tag_sequence = ['O'] + tag_sequence + ['O']

    for i in range(1, len(tag_sequence)+1):
        prev, curr, succ = ext_tag_sequence[i-1:i+2]

        in_ongoing_annotation = prev[0] in ['B', 'I']
        in_ending_annotation  = prev[0] in ['E', 'S']

        starts_annotation = curr[0] in ['B', 'S']
        ends_annotation   = curr[0] in ['E', 'S']
        mid_annotation    = curr[0] in ['I']

        if in_ongoing_annotation:
            # does not allow new annotations in this scheme
            assert mid_annotation or ends_annotation, (i, prev, curr, succ)

            # need same entity type here
            assert prev[2:] == curr[2:], (i, prev, curr, succ)

        elif in_ending_annotation:
            # does not allow ongoing annotations in this scheme
            assert not (mid_annotation or ends_annotation), (i, prev, curr, succ)

for fname, elements in predictions.items():
    all_labels = elements
    new_labels = []
    
    for sid, sent in enumerate(all_labels[0]):
        maj_labels = [Counter() for _ in all_labels[0][sid]]
        for model in all_labels:
            for i, label in enumerate(model[sid]):
                label = label.replace('S-', 'B-').replace('E-', 'I-')
                maj_labels[i][label] += 1

        all_same = True
        tag_sequence = []
        for i, counter in enumerate(maj_labels):
            if len(counter) > 1:
                all_same = False
                tag_sequence.append(resolve_annotation(counter))
            else:
                tag_sequence.append(counter.most_common()[0][0])

        tag_sequence = correct_tag_sequence(tag_sequence)
        check_tags(tag_sequence)
        new_labels.append(tag_sequence)
        
    ids, texts, begins, ends = infos[fname]
    with open(out_path + fname, 'w') as fout:
        for sid, sent in enumerate(texts):
            for tid, token in enumerate(sent):
                fout.write(f'{ids[sid][tid]}\t')
                fout.write(f'{texts[sid][tid]}\t')
                fout.write(f'{begins[sid][tid]}\t')
                fout.write(f'{ends[sid][tid]}\t')
                fout.write(f'{new_labels[sid][tid]}\n')
            fout.write('\n')
        