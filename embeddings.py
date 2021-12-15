# Embeddings for SequenceTagger model.
# This includes custom TransformerWordEmbeddings 
# that perform NER based on token IDs instead of text
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
#
# This source code is based on the flairNLP Project V 0.8 
#   (https://github.com/flairNLP/flair/releases/tag/v0.8)
# Copyright (c) 2018 Zalando SE, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import flair
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence

from typing import List, Union

import torch
    
from transformers import AutoTokenizer,CONFIG_MAPPING


class SubwordTransformerWordEmbeddings(TransformerWordEmbeddings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # The input is given as vocabulary ids
        self.input_is_encoded = True

    def _add_embeddings_to_sentence(self, sentence: Sentence):
        """Match subtokenization to Flair tokenization and extract embeddings from transformers for each token."""
        
        if not hasattr(self, 'max_subtokens_sequence_length'):
            self.max_subtokens_sequence_length = None
            self.allow_long_sentences = False
            self.stride = 0

        # if we also use context, first expand sentence to include context
        if self.context_length > 0:
            # in case of contextualization, we must remember non-expanded sentence
            original_sentence = sentence

            # create expanded sentence and remember context offsets
            expanded_sentence, context_offset = self._expand_sentence_with_context(sentence)

            # overwrite sentence with expanded sentence
            sentence = expanded_sentence
        
        if self.input_is_encoded:
            # if sentence is too long, will be split into multiple parts
            sentence_splits = []
            
            prep_tokens = [token.text.replace('<DOCSTART>', '1') for token in sentence.tokens] # convert <DOCSTART> to <pad>
            prep_tokens = ['1' if not token.strip() else token for token in prep_tokens] # convert empty tokens to <pad>
            encoded_inputs = [0] + [int(token) for token in prep_tokens] + [2] # add <s> and </s> tokens
            token_subtoken_lengths = [1 for x in encoded_inputs] # flair should handle every subword as its own token
            assert len(encoded_inputs) <= 514
            sentence_splits.append(torch.tensor(encoded_inputs, dtype=torch.long))
        
        else:
            raise ValueError('This model only accepts vocabulary ids as input')

        # embed each sentence split
        hidden_states_of_all_splits = []
        for split_number, sentence_split in enumerate(sentence_splits):

            # initialize batch tensors and mask
            input_ids = sentence_split.unsqueeze(0).to(flair.device)

            # propagate gradients if fine-tuning and only during training
            propagate_gradients = self.fine_tune and self.training
            # increase memory effectiveness by skipping all but last sentence split
            if propagate_gradients and self.memory_effective_training and split_number < len(sentence_splits) - 1:
                propagate_gradients = False

            # put encoded batch through transformer model to get all hidden states of all encoder layers
            if propagate_gradients:
                hidden_states = self.model(input_ids)[-1]  # make the tuple a tensor; makes working with it easier.
            else:
                with torch.no_grad():  # deactivate gradients if not necessary
                    hidden_states = self.model(input_ids)[-1]

            # get hidden states as single tensor
            split_hidden_state = torch.stack(hidden_states)[:, 0, ...]
            hidden_states_of_all_splits.append(split_hidden_state)

        # put splits back together into one tensor using overlapping strides
        hidden_states = hidden_states_of_all_splits[0]
        for i in range(1, len(hidden_states_of_all_splits)):
            hidden_states = hidden_states[:, :-1 - self.stride // 2, :]
            next_split = hidden_states_of_all_splits[i]
            next_split = next_split[:, 1 + self.stride // 2:, :]
            hidden_states = torch.cat([hidden_states, next_split], 1)

        subword_start_idx = self.begin_offset

        # for each token, get embedding
        for token_idx, (token, number_of_subtokens) in enumerate(zip(sentence, token_subtoken_lengths)):

            # some tokens have no subtokens at all (if omitted by BERT tokenizer) so return zero vector
            if number_of_subtokens == 0:
                token.set_embedding(self.name, torch.zeros(self.embedding_length))
                continue

            subword_end_idx = subword_start_idx + number_of_subtokens

            subtoken_embeddings: List[torch.FloatTensor] = []

            # get states from all selected layers, aggregate with pooling operation
            for layer in self.layer_indexes:
                current_embeddings = hidden_states[layer][subword_start_idx:subword_end_idx]

                if self.pooling_operation == "first":
                    final_embedding: torch.FloatTensor = current_embeddings[0]

                if self.pooling_operation == "last":
                    final_embedding: torch.FloatTensor = current_embeddings[-1]

                if self.pooling_operation == "first_last":
                    final_embedding: torch.Tensor = torch.cat([current_embeddings[0], current_embeddings[-1]])

                if self.pooling_operation == "mean":
                    all_embeddings: List[torch.FloatTensor] = [
                        embedding.unsqueeze(0) for embedding in current_embeddings
                    ]
                    final_embedding: torch.Tensor = torch.mean(torch.cat(all_embeddings, dim=0), dim=0)

                subtoken_embeddings.append(final_embedding)

            # use layer mean of embeddings if so selected
            if self.layer_mean and len(self.layer_indexes) > 1:
                sm_embeddings = torch.mean(torch.stack(subtoken_embeddings, dim=1), dim=1)
                subtoken_embeddings = [sm_embeddings]

            # set the extracted embedding for the token
            token.set_embedding(self.name, torch.cat(subtoken_embeddings))

            subword_start_idx += number_of_subtokens

        # move embeddings from context back to original sentence (if using context)
        if self.context_length > 0:
            for token_idx, token in enumerate(original_sentence):
                token.set_embedding(self.name, sentence[token_idx + context_offset].get_embedding(self.name))
            sentence = original_sentence
            

    def __setstate__(self, d):
        self.__dict__ = d

        # necessary for reverse compatibility with Flair <= 0.7
        if 'use_scalar_mix' in self.__dict__.keys():
            self.__dict__['layer_mean'] = d['use_scalar_mix']
        if not 'memory_effective_training' in self.__dict__.keys():
            self.__dict__['memory_effective_training'] = True
        if 'pooling_operation' in self.__dict__.keys():
            self.__dict__['subtoken_pooling'] = d['pooling_operation']
        if not 'context_length' in self.__dict__.keys():
            self.__dict__['context_length'] = 0
        if 'use_context' in self.__dict__.keys():
            self.__dict__['context_length'] = 64 if self.__dict__['use_context'] == True else 0

        if not 'context_dropout' in self.__dict__.keys():
            self.__dict__['context_dropout'] = 0.5
        if not 'respect_document_boundaries' in self.__dict__.keys():
            self.__dict__['respect_document_boundaries'] = True
        if not 'memory_effective_training' in self.__dict__.keys():
            self.__dict__['memory_effective_training'] = True
        if not 'base_model_name' in self.__dict__.keys():
            self.__dict__['base_model_name'] = self.__dict__['name'].split('transformer-word-')[-1]

        # special handling for deserializing transformer models
        if "config_state_dict" in d:

            # load transformer model
            if "model_type" not in d["config_state_dict"]:
                d["config_state_dict"]["model_type"] = "bert" # default
                try:
                    for entry in d["config_state_dict"]['architectures']:
                        if 'roberta' in entry.lower():
                            d["config_state_dict"]["model_type"] = "roberta"
                except:
                    pass
            print(d["config_state_dict"])
            print('model_type: ' + d["config_state_dict"]["model_type"])
                    
            config_class = CONFIG_MAPPING[d["config_state_dict"]["model_type"]]
            loaded_config = config_class.from_dict(d["config_state_dict"])

            # constructor arguments
            layers = ','.join([str(idx) for idx in self.__dict__['layer_indexes']])

            # re-initialize transformer word embeddings with constructor arguments
            print('Load transformer from: ' + self.__dict__['base_model_name'])
            embedding = SubwordTransformerWordEmbeddings(
                model=self.__dict__['base_model_name'],
                layers=layers,
                subtoken_pooling=self.__dict__['subtoken_pooling'],
                use_context=self.__dict__['context_length'],
                layer_mean=self.__dict__['layer_mean'],
                fine_tune=self.__dict__['fine_tune'],
                allow_long_sentences=self.__dict__['allow_long_sentences'],
                respect_document_boundaries=self.__dict__['respect_document_boundaries'],
                memory_effective_training=self.__dict__['memory_effective_training'],
                context_dropout=self.__dict__['context_dropout'],

                config=loaded_config,
                state_dict=d["model_state_dict"],
            )

            # I have no idea why this is necessary, but otherwise it doesn't work
            for key in embedding.__dict__.keys():
                self.__dict__[key] = embedding.__dict__[key]

        else:

            # reload tokenizer to get around serialization issues
            model_name = self.__dict__['name'].split('transformer-word-')[-1]
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            except:
                pass

            self.tokenizer = tokenizer