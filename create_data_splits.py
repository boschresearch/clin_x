# Part 2 of our training pipeline:
# Create the random datasplit
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
import numpy as np
import torch
import torch.nn.functional as F
import os

from datasplits import perform_pca, perform_kmeans, perform_kmeans_size, create_random_datasplits
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--train_files', type=str, help='Path to input train files in bio format')
parser.add_argument('--dev_files', type=str, default=None, help='Path to input dev files. Will be concatenated to train if given')
parser.add_argument('--other_files', type=str, default=None, help='Path to input other files. Will be concatenated to train if given')
parser.add_argument('--output_dir', type=str, default=None, help='Path to output directory')
parser.add_argument('--num_splits', type=int, default=5, help='Number of splits')
parser.add_argument('--n_components', type=int, default=5, help='Number of PCA components before clustering')
parser.add_argument('--method', '-x', type=str, default='strategic', choices=['strategic', 'random'])
parser.add_argument('--model_path', '-m', type=str, default='xlm-roberta-large')
parser.add_argument('--seed', type=int, default=555, help='Random seed')
args = parser.parse_args()


def doc_to_vec(fname, model, tokenizer, maxlen=510):
    """Read text documents and get vector representations. 
    The vector is based on the averaged vector for each sentence
    from the huggingface transformer model. """
    #print(fname)
    sentences = []
    with open(fname, 'r') as fin:
        content = fin.read().splitlines()
        
        cur_tokens = []
        for line in content[1:]: # skip the first line
            line = line.strip()
            if line:
                s = line.split('\t')
                if len(s) == 5:
                    idx, text, begin, end, label = s
                    #cur_tokens.append(int(idx))
                    cur_tokens.append(text)
                elif len(s) == 4:
                    text, begin, end, label = s
                    cur_tokens.append(text)
                elif len(s) == 2:
                    text, label = s
                    cur_tokens.append(text)
            elif len(cur_tokens) > 0:
                #sentences.append([0,] + cur_tokens + [2,])
                sentences.append(cur_tokens)
                cur_tokens = []
            # cut long sentences
            if len(cur_tokens) >= maxlen:
                #sentences.append([0,] + cur_tokens + [2,])
                sentences.append(cur_tokens)
                cur_tokens = []
        if len(cur_tokens) > 0:
            #sentences.append([0,] + cur_tokens + [2,])
            sentences.append(cur_tokens)
    
    doc_vec = torch.zeros(1024).cuda()
    num_subwords = sum([len(sent) for sent in sentences])
    max_len = max([len(sent) for sent in sentences])
    
    input_ids = [tokenizer(' '.join(x))['input_ids'] for x in sentences]
    input_ids = [torch.tensor(sent, dtype=torch.long) for sent in input_ids]
    input_ids = [F.pad(x, pad=(0, max_len - x.numel()), mode='constant', value=1) for x in input_ids]
    input_ids = torch.stack(input_ids).long().cuda() 
    
    with torch.no_grad():  # deactivate gradients if not necessary
        outputs = model(input_ids)[0]
        #print(outputs.shape)
    # get hidden states as single tensor
    for sentence in outputs:
        for token in sentence:
            doc_vec += token
    doc_vec /= num_subwords
    return num_subwords, doc_vec

#
# Read the file and create datasplits
#

print('Create splits for files in ' + args.train_files)
all_files = [args.train_files]
if args.dev_files is not None:
    print('--------- and for files in ' + args.dev_files)
    all_files.append(args.dev_files)
        
        
args.output_dir = args.output_dir if args.output_dir.endswith('/') else args.output_dir + '/'
if os.path.exists(args.output_dir + 'doc_vectors.txt'):
    sentence_vectors_dict = {}
    print('Reload document vectors from file')
    with open(args.output_dir + 'doc_vectors.txt', 'r') as fin:
        for line in fin:
            line = line.strip()
            if len(line) > 0:
                fname, length, vector = line.split('\t')
                vector = [float(value) for value in vector[1:-1].split(', ')]
                vector = np.array(vector)
                sentence_vectors_dict[fname] = (length, vector)
    
else:
    print('Compute new document vectors')
    model = AutoModel.from_pretrained(args.model_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    sentence_vectors_dict = {}
    for parent_dir in all_files:
        for fname in tqdm(os.listdir(parent_dir)):
            if fname.endswith('.bio'):
                num_subwords, doc_vec = doc_to_vec(parent_dir + fname, model, tokenizer)
                sentence_vectors_dict[parent_dir + fname] = (num_subwords, doc_vec)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.output_dir + 'doc_vectors.txt', 'w') as fout:
        for fname, vector in sentence_vectors_dict.items():
            fout.write(fname)
            fout.write('\t')
            fout.write(str(vector[0]))
            fout.write('\t')
            fout.write(str([v.item() for v in vector[1]]))
            fout.write('\n')
        
# Number of Dimensions for PCA
n_components = args.n_components

# Clustering Algorithm and Number of Clusters
# options: 'distribution_sensitive_kmeans', 'same_size_kmeans', 'kmeans', 'randomized'
clustering_algorithm = 'same_size_kmeans' if args.method == 'strategic' else 'randomized'
num_clusters = args.num_splits

# Options for the Size and Distribution Sensitive K-Means algorithm
n_jobs = 8
print("Number of instances to be clustered:", len(sentence_vectors_dict.values()))


# step (1): centering and scaling
doc_vectors, doc_ids = [], []
for fname, vector in sentence_vectors_dict.items():
    doc_ids.append(fname)
    try:
        doc_vectors.append(vector[1].cpu().numpy())
    except:
        doc_vectors.append(vector[1])

sentence_vectors = np.array(doc_vectors)
scaler = StandardScaler().fit(sentence_vectors)
sentence_vectors = scaler.transform(sentence_vectors)

# step (2): perform PCA
sentence_vectors, explained_variance_ratio, explained_variance = perform_pca(sentence_vectors, n_components)

cluster_sizes = [0 for i in range(num_clusters)]
cluster_to_fill = 0
while sum(cluster_sizes) < len(doc_vectors):
    cluster_sizes[cluster_to_fill] += 1
    cluster_to_fill = (cluster_to_fill + 1) % num_clusters
print('Clustering of size:')
print(cluster_sizes)
    
max_ids_labels = [
    np.array(cluster_sizes),
]

if clustering_algorithm == "kmeans":
    labels_clustering, centroids = perform_kmeans(sentence_vectors, num_clusters)

elif clustering_algorithm == "same_size_kmeans":
    labels_clustering, centroids = perform_kmeans_size(sentence_vectors, num_clusters, n_jobs, max_ids_labels = max_ids_labels)
    
elif clustering_algorithm == "randomized":
    labels_clustering = create_random_datasplits(sentence_vectors, num_clusters)

# add 1, otherwise cluster labels would start at 0 (design choice)
labels_clustering += 1

for n in range(1, num_clusters+1):
    with open(args.output_dir + args.method + '_split_' + str(n) + '.txt', 'w') as fout:
        pass  # clean file

with open(args.output_dir + 'cluster_list_' + args.method + '.txt', 'w') as fout_list:
    for doc_id, cluster in zip(doc_ids, labels_clustering):
        fout_list.write(f'{doc_id}\t{int(cluster)}\n')
        
        with open(args.output_dir + args.method + '_split_' + str(int(cluster)) + '.txt', 'a') as fout_doc:
            with open(doc_id, 'r') as fin_doc:
                content = fin_doc.read()
            fout_doc.write(content + '\n\n')