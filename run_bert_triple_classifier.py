# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import multiprocessing as mp
import math
import csv
import json
import logging
import os
import random
import sys
import shutil

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from functools import partial

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn import metrics

# Optional plotting
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _PLOT_AVAILABLE = True
except Exception:
    _PLOT_AVAILABLE = False

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule


#os.environ['CUDA_VISIBLE_DEVICES']= '6'
#torch.backends.cudnn.deterministic = True

logger = logging.getLogger(__name__)

# Global toggles for multiprocessing in example building
_MP_BUILD_WORKERS = 1
_MP_BUILD_CHUNKSIZE = 1000

# ------- Helpers for multiprocessing example building -------
# Read-only globals shared via fork (copy-on-write). Avoids pickling huge data into each worker.
_BW_ENT2TEXT = None
_BW_REL2TEXT = None
_BW_ENTITIES = None
_BW_LINES_SET = None
_BW_SET_TYPE = None

def _init_build_worker_noop():
    return

def _build_examples_for_line(index_and_line):
    """Build one or more InputExample objects from a single line (and its index).

    Returns a list to allow returning both positive and corrupted negative examples for train split.
    """
    i, line = index_and_line
    set_type = _BW_SET_TYPE
    ent2text = _BW_ENT2TEXT
    rel2text = _BW_REL2TEXT
    entities = _BW_ENTITIES
    lines_str_set = _BW_LINES_SET

    head_ent_text = ent2text[line[0]]
    tail_ent_text = ent2text[line[2]]
    relation_text = rel2text[line[1]]

    if set_type == "dev" or set_type == "test":
        triple_label = line[3]
        label = "1" if triple_label == "1" else "0"
        guid = "%s-%s" % (set_type, i)
        text_a = head_ent_text
        text_b = relation_text
        text_c = tail_ent_text
        return [InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label=label)]

    # train split: create positive + one corrupted negative
    examples = []
    guid = "%s-%s" % (set_type, i)
    text_a = head_ent_text
    text_b = relation_text
    text_c = tail_ent_text
    examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label="1"))

    rnd = random.random()
    guid = "%s-%s" % (set_type + "_corrupt", i)
    if rnd <= 0.5:
        # corrupt head
        tmp_head = ''
        while True:
            tmp_ent_list = set(entities)
            tmp_ent_list.remove(line[0])
            tmp_ent_list = list(tmp_ent_list)
            tmp_head = random.choice(tmp_ent_list)
            tmp_triple_str = tmp_head + '\t' + line[1] + '\t' + line[2]
            if tmp_triple_str not in lines_str_set:
                break
        tmp_head_text = ent2text[tmp_head]
        examples.append(InputExample(guid=guid, text_a=tmp_head_text, text_b=text_b, text_c=text_c, label="0"))
    else:
        # corrupt tail
        tmp_tail = ''
        while True:
            tmp_ent_list = set(entities)
            tmp_ent_list.remove(line[2])
            tmp_ent_list = list(tmp_ent_list)
            tmp_tail = random.choice(tmp_ent_list)
            tmp_triple_str = line[0] + '\t' + line[1] + '\t' + tmp_tail
            if tmp_triple_str not in lines_str_set:
                break
        tmp_tail_text = ent2text[tmp_tail]
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=tmp_tail_text, label="0"))

    return examples


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            text_c: (Optional) string. The untokenized text of the third sequence.
            Only must be specified for sequence triple tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class KGProcessor(DataProcessor):
    """Processor for knowledge graph data set."""
    def __init__(self):
        self.labels = set()
    
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_dir)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", data_dir)

    def get_test_examples(self, data_dir):
      """See base class."""
      return self._create_examples(
          self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", data_dir)

    def get_relations(self, data_dir):
        """Gets all labels (relations) in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "relations.txt"), 'r') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip())
        return relations

    def get_labels(self, data_dir):
        """Gets all labels (0, 1) for triples in the knowledge graph."""
        return ["0", "1"]

    def get_entities(self, data_dir):
        """Gets all entities in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "entities.txt"), 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip())
        return entities

    def get_train_triples(self, data_dir):
        """Gets training triples."""
        return self._read_tsv(os.path.join(data_dir, "train.tsv"))

    def get_dev_triples(self, data_dir):
        """Gets validation triples."""
        return self._read_tsv(os.path.join(data_dir, "dev.tsv"))

    def get_test_triples(self, data_dir):
        """Gets test triples."""
        return self._read_tsv(os.path.join(data_dir, "test.tsv"))

    def _create_examples(self, lines, set_type, data_dir):
        """Creates examples for the training and dev sets."""
        # entity to text
        ent2text = {}
        with open(os.path.join(data_dir, "entity2text.txt"), 'r') as f:
            ent_lines = f.readlines()
            logger.info("Loading entities from entity2text.txt (%d lines)", len(ent_lines))
            for line in tqdm(ent_lines, desc="Loading entities", unit="ent"):
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    ent2text[temp[0]] = temp[1]

        entities = list(ent2text.keys())

        rel2text = {}
        with open(os.path.join(data_dir, "relation2text.txt"), 'r') as f:
            rel_lines = f.readlines()
            logger.info("Loading relations from relation2text.txt (%d lines)", len(rel_lines))
            for line in tqdm(rel_lines, desc="Loading relations", unit="rel"):
                temp = line.strip().split('\t')
                rel2text[temp[0]] = temp[1]      

        lines_str_set = set(['\t'.join(line) for line in lines])
        examples = []
        logger.info("Processing %s split with %d triples", set_type, len(lines))

        # Optional multiprocessing path for building examples (esp. for large train sets)
        if set_type == "train" and _MP_BUILD_WORKERS > 1:
            logger.info("Building examples with multiprocessing: workers=%d, chunksize=%d", _MP_BUILD_WORKERS, _MP_BUILD_CHUNKSIZE)
            # Set globals in parent so forked workers inherit without pickling
            global _BW_ENT2TEXT, _BW_REL2TEXT, _BW_ENTITIES, _BW_LINES_SET, _BW_SET_TYPE
            _BW_ENT2TEXT = ent2text
            _BW_REL2TEXT = rel2text
            _BW_ENTITIES = entities
            _BW_LINES_SET = lines_str_set
            _BW_SET_TYPE = set_type

            ctx = mp.get_context("fork")
            with ctx.Pool(processes=_MP_BUILD_WORKERS, initializer=_init_build_worker_noop) as pool:
                # enumerate lines to preserve guids similar to single-process path
                for ex_list in tqdm(pool.imap(_build_examples_for_line, enumerate(lines), chunksize=max(1, _MP_BUILD_CHUNKSIZE)),
                                    total=len(lines), desc=f"Building {set_type} examples (mp)", unit="ex"):
                    if ex_list:
                        examples.extend(ex_list)
            return examples

        for (i, line) in enumerate(tqdm(lines, desc=f"Building {set_type} examples", unit="ex")):
            
            head_ent_text = ent2text[line[0]]
            tail_ent_text = ent2text[line[2]]
            relation_text = rel2text[line[1]]

            if set_type == "dev" or set_type == "test":
                triple_label = line[3]
                if triple_label == "1":
                    label = "1"
                else:
                    label = "0"

                guid = "%s-%s" % (set_type, i)
                text_a = head_ent_text
                text_b = relation_text
                text_c = tail_ent_text 
                self.labels.add(label)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = text_c, label=label))
                
            elif set_type == "train":
                guid = "%s-%s" % (set_type, i)
                text_a = head_ent_text
                text_b = relation_text
                text_c = tail_ent_text 
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = text_c, label="1"))

                rnd = random.random()
                guid = "%s-%s" % (set_type + "_corrupt", i)
                if rnd <= 0.5:
                    # corrupting head
                    tmp_head = ''
                    while True:
                        tmp_ent_list = set(entities)
                        tmp_ent_list.remove(line[0])
                        tmp_ent_list = list(tmp_ent_list)
                        tmp_head = random.choice(tmp_ent_list)
                        tmp_triple_str = tmp_head + '\t' + line[1] + '\t' + line[2]
                        if tmp_triple_str not in lines_str_set:
                            break                    
                    tmp_head_text = ent2text[tmp_head]
                    examples.append(
                        InputExample(guid=guid, text_a=tmp_head_text, text_b=text_b, text_c = text_c, label="0"))       
                else:
                    # corrupting tail
                    tmp_tail = ''
                    while True:
                        tmp_ent_list = set(entities)
                        tmp_ent_list.remove(line[2])
                        tmp_ent_list = list(tmp_ent_list)
                        tmp_tail = random.choice(tmp_ent_list)
                        tmp_triple_str = line[0] + '\t' + line[1] + '\t' + tmp_tail
                        if tmp_triple_str not in lines_str_set:
                            break
                    tmp_tail_text = ent2text[tmp_tail]
                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = tmp_tail_text, label="0"))                                                  
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, print_info = True):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(tqdm(examples, desc="Converting examples to features", unit="ex")):
        if ex_index % 10000 == 0 and print_info:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        tokens_c = None

        if example.text_b and example.text_c:
            tokens_b = tokenizer.tokenize(example.text_b)
            tokens_c = tokenizer.tokenize(example.text_c)
            # Modifies `tokens_a`, `tokens_b` and `tokens_c`in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP], [SEP] with "- 4"
            #_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        # (c) for sequence triples:
        #  tokens: [CLS] Steve Jobs [SEP] founded [SEP] Apple Inc .[SEP]
        #  type_ids: 0 0 0 0 1 1 0 0 0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence or the third sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
        if tokens_c:
            tokens += tokens_c + ["[SEP]"]
            segment_ids += [0] * (len(tokens_c) + 1)        

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        if ex_index < 5 and print_info:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


# -------- Multiprocessing variant for faster tokenization on large datasets --------
_G_tokenizer = None
_G_label_map = None
_G_max_len = None
_G_show_debug = False

def _init_feature_worker(tokenizer_path, do_lower_case, label_list, max_seq_length, show_debug=False):
    """Initializer for multiprocessing workers: load tokenizer and constants once per process."""
    global _G_tokenizer, _G_label_map, _G_max_len, _G_show_debug
    _G_tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=do_lower_case)
    _G_label_map = {label: i for i, label in enumerate(label_list)}
    _G_max_len = max_seq_length
    _G_show_debug = show_debug

def _example_to_feature_worker(example):
    """Convert a single example to features using worker globals."""
    # Tokenize
    tokens_a = _G_tokenizer.tokenize(example.text_a)
    tokens_b = _G_tokenizer.tokenize(example.text_b) if example.text_b else None
    tokens_c = _G_tokenizer.tokenize(example.text_c) if example.text_c else None

    if tokens_b is not None and tokens_c is not None:
        _truncate_seq_triple(tokens_a, tokens_b, tokens_c, _G_max_len - 4)
    else:
        if len(tokens_a) > _G_max_len - 2:
            tokens_a = tokens_a[:(_G_max_len - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)
    if tokens_c:
        tokens += tokens_c + ["[SEP]"]
        segment_ids += [0] * (len(tokens_c) + 1)

    input_ids = _G_tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding = [0] * (_G_max_len - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    label_id = _G_label_map[example.label]
    return InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
    )

def convert_examples_to_features_mp(examples, label_list, max_seq_length, tokenizer_path, do_lower_case, num_workers=1, chunksize=100, show_debug=False):
    """Multiprocessing version of convert_examples_to_features.

    Falls back to single-process when num_workers <= 1.
    """
    if num_workers <= 1:
        # Reuse existing single-process path by constructing a tokenizer here
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=do_lower_case)
        return convert_examples_to_features(examples, label_list, max_seq_length, tokenizer)

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers,
                  initializer=_init_feature_worker,
                  initargs=(tokenizer_path, do_lower_case, label_list, max_seq_length, show_debug)) as pool:
        features = []
        for feat in tqdm(pool.imap(_example_to_feature_worker, examples, chunksize=max(1, chunksize)),
                         total=len(examples), desc="Converting examples to features (mp)", unit="ex"):
            features.append(feat)
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence triple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        # Pick the longest non-empty sequence to truncate; on ties, still ensure non-empty
        lens_and_seqs = [
            (len(tokens_a), tokens_a),
            (len(tokens_b), tokens_b),
            (len(tokens_c), tokens_c),
        ]
        lens_and_seqs.sort(key=lambda x: x[0], reverse=True)
        # Pop from the first non-empty sequence
        for seq_len, seq in lens_and_seqs:
            if seq_len > 0:
                seq.pop()
                break
        else:
            # All sequences are empty (shouldn't happen if total_length > max_length), safeguard to avoid IndexError
            break

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "kg":
        # Include F1 for binary triple classification
        return {"acc": simple_accuracy(preds, labels), "f1": f1_score(labels, preds)}
    else:
        raise KeyError(task_name)


def softmax_np(x, axis=1):
    """Numerically stable softmax for numpy arrays."""
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def _ensure_legacy_config_filename(model_dir):
    """If only config.json exists (new HF layout), copy to bert_config.json for legacy loader."""
    try:
        if os.path.isdir(model_dir):
            new_cfg = os.path.join(model_dir, "config.json")
            old_cfg = os.path.join(model_dir, CONFIG_NAME)  # usually bert_config.json in legacy lib
            if os.path.isfile(new_cfg) and (not os.path.isfile(old_cfg)):
                shutil.copy2(new_cfg, old_cfg)
    except Exception as e:
        logger.warning("Legacy config filename adaptation failed for %s: %s", str(model_dir), str(e))

def _has_checkpoint(dir_path):
    try:
        return os.path.isfile(os.path.join(dir_path, WEIGHTS_NAME)) and os.path.isfile(os.path.join(dir_path, CONFIG_NAME))
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--disable_data_parallel', action='store_true', help='Disable torch.nn.DataParallel even when multiple GPUs are visible')
    parser.add_argument('--num_workers', type=int, default=0, help='Processes for parallel feature conversion (0/1 = single process).')
    parser.add_argument('--mp_chunksize', type=int, default=100, help='Chunksize for multiprocessing imap.')
    parser.add_argument('--build_workers', type=int, default=1, help='Processes for building training examples (train split).')
    parser.add_argument('--build_chunksize', type=int, default=1000, help='Chunksize for multiprocessing in example building.')
    parser.add_argument('--eval_every_steps', type=int, default=0, help='Evaluate on dev set every N training steps (0 disables).')
    parser.add_argument('--plot_metrics', action='store_true', help='Plot AUC/ACC/F1 curves to output_dir when evaluating.')
    parser.add_argument('--metrics_file_prefix', type=str, default='metrics', help='Prefix for per-step metrics files in output_dir.')
    parser.add_argument('--smooth_alpha', type=float, default=0.0, help='EMA smoothing factor for plots (0 disables).')
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "kg": KGProcessor,
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    # args.seed = random.randint(1, 200)  # Commented out: random seed makes debugging harder
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    # Wire global toggles for example building MP
    global _MP_BUILD_WORKERS, _MP_BUILD_CHUNKSIZE
    _MP_BUILD_WORKERS = max(1, int(getattr(args, 'build_workers', 1)))
    _MP_BUILD_CHUNKSIZE = max(1, int(getattr(args, 'build_chunksize', 1000)))

    label_list = processor.get_labels(args.data_dir)
    num_labels = len(label_list)

    entity_list = processor.get_entities(args.data_dir)
    #print(entity_list)

    # Ensure legacy config filename if a local directory is used
    _ensure_legacy_config_filename(args.bert_model)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = 0
    # Pre-build dev eval dataloader if we will evaluate periodically or if do_eval requested
    eval_dataloader = None
    eval_label_ids_numpy = None

    def _build_eval_dataloader():
        nonlocal eval_dataloader, eval_label_ids_numpy
        eval_examples = processor.get_dev_examples(args.data_dir)
        if getattr(args, 'num_workers', 0) and args.num_workers > 1:
            eval_features = convert_examples_to_features_mp(
                eval_examples, label_list, args.max_seq_length,
                args.bert_model, args.do_lower_case,
                num_workers=args.num_workers, chunksize=getattr(args, 'mp_chunksize', 100)
            )
        else:
            eval_features = convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_label_ids_numpy = all_label_ids.numpy()
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    def _evaluate_current_model(current_model):
        current_model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        loss_fct_local = CrossEntropyLoss()
        with torch.no_grad():
            for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                logits = current_model(input_ids, segment_ids, input_mask, labels=None)
                tmp_eval_loss = loss_fct_local(logits.view(-1, num_labels), label_ids.view(-1))
                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                logits_np = logits.detach().cpu().numpy()
                if len(preds) == 0:
                    preds.append(logits_np)
                else:
                    preds[0] = np.append(preds[0], logits_np, axis=0)
        eval_loss = eval_loss / max(1, nb_eval_steps)
        logits_np = preds[0]
        pred_labels = np.argmax(logits_np, axis=1)
        result_core = compute_metrics(task_name, pred_labels, eval_label_ids_numpy)
        result = {
            "eval_loss": eval_loss,
            "acc": float(result_core["acc"]),
            "f1": float(result_core["f1"]),
        }
        if num_labels == 2:
            try:
                prob_pos = softmax_np(logits_np, axis=1)[:, 1]
                result['auc'] = float(metrics.roc_auc_score(eval_label_ids_numpy, prob_pos))
            except Exception as e:
                logger.warning("AUC computation failed: %s", str(e))
        current_model.train()
        return result

    metrics_history = []
    metrics_csv_path = None
    metrics_jsonl_path = None

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        steps_per_epoch = max(1, int(math.ceil(len(train_examples) / float(max(1, args.train_batch_size)) / float(max(1, args.gradient_accumulation_steps)))))
        num_train_optimization_steps = int(steps_per_epoch * float(args.num_train_epochs))
        if args.local_rank != -1:
            world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            num_train_optimization_steps = max(1, num_train_optimization_steps // max(1, world_size))

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    _ensure_legacy_config_filename(args.bert_model)
    model = BertForSequenceClassification.from_pretrained(args.bert_model,
              cache_dir=cache_dir,
              num_labels=num_labels)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1 and (not args.disable_data_parallel):
        # Choose number of GPUs that divides the effective batch size to avoid empty shards
        max_devices = min(n_gpu, max(1, args.train_batch_size))
        best_k = 1
        for k in range(max_devices, 0, -1):
            if args.train_batch_size % k == 0:
                best_k = k
                break
        if best_k == 1:
            logger.warning(
                "Using single GPU: effective train_batch_size=%d is not divisible by any GPU count up to %d.",
                args.train_batch_size, n_gpu
            )
        else:
            device_ids = list(range(best_k))
            logger.info("Enabling DataParallel on %d GPUs: %s", best_k, device_ids)
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        #model = torch.nn.parallel.data_parallel(model)
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                             t_total=num_train_optimization_steps)        

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:

        # Convert to features (optionally with multiprocessing)
        if getattr(args, 'num_workers', 0) and args.num_workers > 1:
            logger.info("Using multiprocessing for feature conversion: workers=%d, chunksize=%d", args.num_workers, getattr(args, 'mp_chunksize', 100))
            train_features = convert_examples_to_features_mp(
                train_examples, label_list, args.max_seq_length,
                args.bert_model, args.do_lower_case,
                num_workers=args.num_workers, chunksize=getattr(args, 'mp_chunksize', 100)
            )
        else:
            train_features = convert_examples_to_features(
                train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        # With multiple GPUs, drop last incomplete batch to avoid empty shard on some device
        drop_last_batches = n_gpu > 1
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=drop_last_batches)

        # Prepare eval dataloader if periodic eval is enabled
        if int(getattr(args, 'eval_every_steps', 0)) > 0:
            _build_eval_dataloader()
            metrics_csv_path = os.path.join(args.output_dir, f"{args.metrics_file_prefix}_history.csv")
            metrics_jsonl_path = os.path.join(args.output_dir, f"{args.metrics_file_prefix}_history.jsonl")
            # Write CSV header
            try:
                with open(metrics_csv_path, 'w') as f:
                    f.write('step,eval_loss,acc,f1,auc\n')
            except Exception as e:
                logger.warning("Failed to init metrics CSV: %s", str(e))

        model.train()
        #print(model)
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                # define a new function to compute loss values for both output_modes
                logits = model(input_ids, segment_ids, input_mask, labels=None)
                #print(logits, logits.shape)

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step/num_train_optimization_steps,
                                                                                 args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    # Periodic evaluation
                    if int(getattr(args, 'eval_every_steps', 0)) > 0 and (global_step % int(args.eval_every_steps) == 0):
                        if eval_dataloader is None:
                            _build_eval_dataloader()
                        metrics_result = _evaluate_current_model(model)
                        metrics_result['step'] = int(global_step)
                        metrics_history.append(metrics_result)
                        # Append to CSV/JSONL
                        try:
                            if metrics_csv_path:
                                with open(metrics_csv_path, 'a') as f:
                                    f.write('{},{:.6f},{:.6f},{:.6f},{}\n'.format(
                                        metrics_result['step'],
                                        metrics_result.get('eval_loss', float('nan')),
                                        metrics_result.get('acc', float('nan')),
                                        metrics_result.get('f1', float('nan')),
                                        ('{:.6f}'.format(metrics_result['auc']) if 'auc' in metrics_result else '')
                                    ))
                            if metrics_jsonl_path:
                                with open(metrics_jsonl_path, 'a') as jf:
                                    jf.write(json.dumps(metrics_result) + "\n")
                        except Exception as e:
                            logger.warning("Failed to write metrics: %s", str(e))
        # Plot if requested and available
        if getattr(args, 'plot_metrics', False) and _PLOT_AVAILABLE and len(metrics_history) > 0:
            try:
                steps = [m['step'] for m in metrics_history]
                # Smoothing helper (EMA, robust to NaN)
                def _ema(values, alpha):
                    if not values or alpha <= 0.0:
                        return values
                    s_vals = []
                    prev = None
                    for v in values:
                        if v is None or (isinstance(v, float) and (np.isnan(v))):
                            s_vals.append(prev if prev is not None else np.nan)
                            continue
                        prev = v if prev is None else (alpha * v + (1.0 - alpha) * prev)
                        s_vals.append(prev)
                    return s_vals
                alpha = max(0.0, min(1.0, float(getattr(args, 'smooth_alpha', 0.0))))

                # ACC
                acc_vals = [m.get('acc') for m in metrics_history]
                acc_plot = _ema(acc_vals, alpha)
                plt.figure()
                plt.plot(steps, acc_plot, label='ACC (EMA{:.2f})'.format(alpha) if alpha > 0 else 'ACC')
                plt.xlabel('Step')
                plt.ylabel('ACC')
                plt.title('Validation ACC over Steps')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, 'acc_curve.png'))
                plt.close()
                # F1
                f1_vals = [m.get('f1') for m in metrics_history]
                f1_plot = _ema(f1_vals, alpha)
                plt.figure()
                plt.plot(steps, f1_plot, label='F1 (EMA{:.2f})'.format(alpha) if alpha > 0 else 'F1', color='orange')
                plt.xlabel('Step')
                plt.ylabel('F1')
                plt.title('Validation F1 over Steps')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, 'f1_curve.png'))
                plt.close()
                # AUC (if exists)
                if any(('auc' in m) for m in metrics_history):
                    auc_vals = [m.get('auc', np.nan) for m in metrics_history]
                    auc_plot = _ema(auc_vals, alpha)
                    plt.figure()
                    plt.plot(steps, auc_plot, label='AUC (EMA{:.2f})'.format(alpha) if alpha > 0 else 'AUC', color='green')
                    plt.xlabel('Step')
                    plt.ylabel('AUC')
                    plt.title('Validation AUC over Steps')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.output_dir, 'auc_curve.png'))
                    plt.close()
            except Exception as e:
                logger.warning("Plotting metrics failed: %s", str(e))
            print("Training loss: ", tr_loss, nb_tr_examples)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        # Prefer checkpoint in output_dir if present; otherwise fall back to original bert_model
        load_dir = args.output_dir if _has_checkpoint(args.output_dir) else args.bert_model
        _ensure_legacy_config_filename(load_dir)
        model = BertForSequenceClassification.from_pretrained(load_dir, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(load_dir, do_lower_case=args.do_lower_case)
    else:
        model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        
        eval_examples = processor.get_dev_examples(args.data_dir)
        if getattr(args, 'num_workers', 0) and args.num_workers > 1:
            eval_features = convert_examples_to_features_mp(
                eval_examples, label_list, args.max_seq_length,
                args.bert_model, args.do_lower_case,
                num_workers=args.num_workers, chunksize=getattr(args, 'mp_chunksize', 100)
            )
        else:
            eval_features = convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        
        # Load a trained model and vocabulary that you have fine-tuned
        _ensure_legacy_config_filename(args.output_dir)
        model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(device)

        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)

            # create eval loss and other metric required by the task
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            print(label_ids.view(-1))
            
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        logits_np = preds[0]

        pred_labels = np.argmax(logits_np, axis=1)
        result = compute_metrics(task_name, pred_labels, all_label_ids.numpy())
        # AUC requires probability scores for the positive class
        if num_labels == 2:
            try:
                prob_pos = softmax_np(logits_np, axis=1)[:, 1]
                result['auc'] = metrics.roc_auc_score(all_label_ids.numpy(), prob_pos)
            except Exception as e:
                logger.warning("AUC computation failed: %s", str(e))
        loss = tr_loss/nb_tr_steps if args.do_train else None

        result['eval_loss'] = eval_loss
        result['global_step'] = global_step
        result['loss'] = loss

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        # Append final eval metrics to CSV/JSONL and regenerate plots from CSV (to include final point)
        try:
            metrics_csv_path = os.path.join(args.output_dir, f"{getattr(args, 'metrics_file_prefix', 'metrics')}_history.csv")
            metrics_jsonl_path = os.path.join(args.output_dir, f"{getattr(args, 'metrics_file_prefix', 'metrics')}_history.jsonl")
            # Ensure CSV with header exists
            if not os.path.isfile(metrics_csv_path):
                with open(metrics_csv_path, 'w') as f:
                    f.write('step,eval_loss,acc,f1,auc\n')
            with open(metrics_csv_path, 'a') as f:
                f.write('{},{:.6f},{:.6f},{:.6f},{}\n'.format(
                    int(global_step),
                    float(result.get('eval_loss', float('nan'))),
                    float(result.get('acc', float('nan'))),
                    float(result.get('f1', float('nan'))),
                    ('{:.6f}'.format(float(result['auc'])) if 'auc' in result else '')
                ))
            if metrics_jsonl_path:
                with open(metrics_jsonl_path, 'a') as jf:
                    jf.write(json.dumps({
                        'step': int(global_step),
                        'eval_loss': float(result.get('eval_loss', float('nan'))),
                        'acc': float(result.get('acc', float('nan'))),
                        'f1': float(result.get('f1', float('nan'))),
                        **({'auc': float(result['auc'])} if 'auc' in result else {})
                    }) + "\n")

            # Regenerate plots based on the full CSV (raw + smoothed)
            if getattr(args, 'plot_metrics', False) and _PLOT_AVAILABLE:
                steps_csv, acc_csv, f1_csv, auc_csv = [], [], [], []
                with open(metrics_csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            steps_csv.append(int(row['step']))
                            acc_csv.append(float(row['acc']))
                            f1_csv.append(float(row['f1']))
                            auc_csv.append(float(row['auc'])) if row.get('auc') not in (None, '', 'nan') else auc_csv.append(np.nan)
                        except Exception:
                            continue
                # Use same EMA smoothing
                def _ema(values, alpha):
                    if not values or float(getattr(args, 'smooth_alpha', 0.0)) <= 0.0:
                        return values
                    alpha_local = max(0.0, min(1.0, float(getattr(args, 'smooth_alpha', 0.0))))
                    s_vals = []
                    prev = None
                    for v in values:
                        if v is None or (isinstance(v, float) and (np.isnan(v))):
                            s_vals.append(prev if prev is not None else np.nan)
                            continue
                        prev = v if prev is None else (alpha_local * v + (1.0 - alpha_local) * prev)
                        s_vals.append(prev)
                    return s_vals

                alpha = max(0.0, min(1.0, float(getattr(args, 'smooth_alpha', 0.0))))

                # ACC plots
                plt.figure()
                plt.plot(steps_csv, _ema(acc_csv, alpha), label='ACC (EMA{:.2f})'.format(alpha) if alpha > 0 else 'ACC')
                plt.scatter(steps_csv, acc_csv, s=6, alpha=0.3, label='ACC raw')
                plt.xlabel('Step'); plt.ylabel('ACC'); plt.title('Validation ACC over Steps'); plt.grid(True); plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, 'acc_curve.png')); plt.close()
                # F1 plots
                plt.figure()
                plt.plot(steps_csv, _ema(f1_csv, alpha), label='F1 (EMA{:.2f})'.format(alpha) if alpha > 0 else 'F1', color='orange')
                plt.scatter(steps_csv, f1_csv, s=6, alpha=0.3, label='F1 raw', color='orange')
                plt.xlabel('Step'); plt.ylabel('F1'); plt.title('Validation F1 over Steps'); plt.grid(True); plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, 'f1_curve.png')); plt.close()
                # AUC plots (if exists)
                if any(not np.isnan(v) for v in auc_csv):
                    plt.figure()
                    plt.plot(steps_csv, _ema(auc_csv, alpha), label='AUC (EMA{:.2f})'.format(alpha) if alpha > 0 else 'AUC', color='green')
                    plt.scatter(steps_csv, auc_csv, s=6, alpha=0.3, label='AUC raw', color='green')
                    plt.xlabel('Step'); plt.ylabel('AUC'); plt.title('Validation AUC over Steps'); plt.grid(True); plt.tight_layout()
                    plt.savefig(os.path.join(args.output_dir, 'auc_curve.png')); plt.close()
        except Exception as e:
            logger.warning("Final metrics append/plot failed: %s", str(e))

    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        train_triples = processor.get_train_triples(args.data_dir)
        dev_triples = processor.get_dev_triples(args.data_dir)
        test_triples = processor.get_test_triples(args.data_dir)
        all_triples = train_triples + dev_triples + test_triples

        all_triples_str_set = set()
        for triple in all_triples:
            triple_str = '\t'.join(triple)
            all_triples_str_set.add(triple_str)

        eval_examples = processor.get_test_examples(args.data_dir)
        if getattr(args, 'num_workers', 0) and args.num_workers > 1:
            eval_features = convert_examples_to_features_mp(
                eval_examples, label_list, args.max_seq_length,
                load_dir, args.do_lower_case,
                num_workers=args.num_workers, chunksize=getattr(args, 'mp_chunksize', 100)
            )
        else:
            eval_features = convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running Prediction *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        # Prefer checkpoint in output_dir if present; otherwise fall back to original bert_model
        load_dir = args.output_dir if _has_checkpoint(args.output_dir) else args.bert_model
        _ensure_legacy_config_filename(load_dir)
        model = BertForSequenceClassification.from_pretrained(load_dir, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(load_dir, do_lower_case=args.do_lower_case)
        model.to(device)
        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Testing"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)

            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        logits_np = preds[0]
        print(logits_np, logits_np.shape)
        
        all_label_ids = all_label_ids.numpy()

        pred_labels = np.argmax(logits_np, axis=1)

        result = compute_metrics(task_name, pred_labels, all_label_ids)
        if num_labels == 2:
            try:
                prob_pos = softmax_np(logits_np, axis=1)[:, 1]
                result['auc'] = metrics.roc_auc_score(all_label_ids, prob_pos)
            except Exception as e:
                logger.warning("AUC computation failed: %s", str(e))
        loss = tr_loss/nb_tr_steps if args.do_train else None

        result['eval_loss'] = eval_loss
        result['global_step'] = global_step
        result['loss'] = loss

        output_eval_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Test results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        print("Triple classification acc is : ")
        print(metrics.accuracy_score(all_label_ids, pred_labels))

if __name__ == "__main__":
    main()
