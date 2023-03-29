# Copyright (c) 2019, Facebook, Inc. and its affiliates. All Rights Reserved
"""
Run SpanBERT on general test examples
Scripts adopted from https://github.com/facebookresearch/SpanBERT and edited by Giannis Karamanolakis
"""

import os
import random
import time
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
#from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from scipy.special import softmax

CLS = "[CLS]"
SEP = "[SEP]"
label_list = ['no_relation', 'per:title', 'org:top_members/employees', 'per:employee_of', 'org:alternate_names', 'org:country_of_headquarters', 'per:countries_of_residence', 'per:age', 'org:city_of_headquarters', 'per:cities_of_residence', 'per:stateorprovinces_of_residence', 'per:origin', 'org:subsidiaries', 'org:parents', 'per:spouse', 'org:stateorprovince_of_headquarters', 'per:children', 'per:other_family', 'org:members', 'per:siblings', 'per:parents', 'per:schools_attended', 'per:date_of_death', 'org:founded_by', 'org:member_of', 'per:cause_of_death', 'org:website', 'org:political/religious_affiliation', 'per:alternate_names', 'org:founded', 'per:city_of_death', 'org:shareholders', 'org:number_of_employees/members', 'per:charges', 'per:city_of_birth', 'per:date_of_birth', 'per:religion', 'per:stateorprovince_of_death', 'per:stateorprovince_of_birth', 'per:country_of_birth', 'org:dissolved', 'per:country_of_death']
special_tokens = {'SUBJ_START': '[unused1]', 'SUBJ_END': '[unused2]', 'OBJ_START': '[unused3]', 'OBJ_END': '[unused4]', 'SUBJ=PERSON': '[unused5]', 'OBJ=TITLE': '[unused6]', 'OBJ=PERSON': '[unused7]', 'OBJ=CITY': '[unused8]', 'SUBJ=ORGANIZATION': '[unused9]', 'OBJ=DATE': '[unused10]', 'OBJ=MISC': '[unused11]', 'OBJ=ORGANIZATION': '[unused12]', 'OBJ=NATIONALITY': '[unused13]', 'OBJ=NUMBER': '[unused14]', 'OBJ=RELIGION': '[unused15]', 'OBJ=URL': '[unused16]', 'OBJ=CAUSE_OF_DEATH': '[unused17]', 'OBJ=COUNTRY': '[unused18]', 'OBJ=DURATION': '[unused19]', 'OBJ=STATE_OR_PROVINCE': '[unused20]', 'OBJ=LOCATION': '[unused21]', 'OBJ=CRIMINAL_CHARGE': '[unused22]', 'OBJ=IDEOLOGY': '[unused23]'} 

class InputExample(object):
    """A single training/test example for span pair classification."""

    def __init__(self, sentence, span1, span2, ner1, ner2):
        self.sentence = sentence
        self.span1 = span1
        self.span2 = span2
        self.ner1 = ner1
        self.ner2 = ner2


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_examples_to_features(examples, max_seq_length, tokenizer, special_tokens):
    """Loads a data file into a list of `InputBatch`s."""

    def create_examples(dataset):
        """Creates examples for the training and dev sets."""
        examples = []
        for example in dataset:
            sentence = example['tokens']
            examples.append(InputExample(
                sentence=example['tokens'],
                ner1=example['subj'][1],
                span1=example['subj'][2],
                ner2=example['obj'][1],
                span2=example['obj'][2]
                ))
        return examples

    def get_special_token(w):
        if w not in special_tokens:
            raise(BaseException("ERROR: did not find special token {} in current dict: {}\n".format(w, special_tokens.keys())))
            #special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
        return special_tokens[w]

    examples = create_examples(examples)
    num_tokens = 0
    num_fit_examples = 0
    num_shown_examples = 0
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = [CLS]
        SUBJECT_START = get_special_token("SUBJ_START")
        SUBJECT_END = get_special_token("SUBJ_END")
        OBJECT_START = get_special_token("OBJ_START")
        OBJECT_END = get_special_token("OBJ_END")
        SUBJECT_NER = get_special_token("SUBJ=%s" % example.ner1)
        OBJECT_NER = get_special_token("OBJ=%s" % example.ner2)

        subj_tokens = []
        obj_tokens = []
        for i, token in enumerate(example.sentence):
            if i == example.span1[0]:
                tokens.append(SUBJECT_NER)
            if i == example.span2[0]:
                tokens.append(OBJECT_NER)
            if (i >= example.span1[0]) and (i <= example.span1[1]):
                for sub_token in tokenizer.tokenize(token):
                    subj_tokens.append(sub_token)
            elif (i >= example.span2[0]) and (i <= example.span2[1]):
                for sub_token in tokenizer.tokenize(token):
                    obj_tokens.append(sub_token)
            else:
                for sub_token in tokenizer.tokenize(token):
                    tokens.append(sub_token)
                    tokens.append(sub_token)
            tokens.append(SEP)
        num_tokens += len(tokens)

        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
        else:
            num_fit_examples += 1

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids
                              ))
    #  print("Average #tokens: %.2f" % (num_tokens * 1.0 / len(examples)))
    #  print("%d (%.2f %%) examples can fit max_seq_length = %d" % (num_fit_examples,
    #      num_fit_examples * 100.0 / len(examples), max_seq_length))
    return features


def predict(model, device, eval_dataloader, verbose=True):
    model.eval()
    preds = []
    for input_ids, input_mask, segment_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    pred_ids = np.argmax(preds[0], axis=1)
    pred_proba = np.max(softmax(preds[0], axis=1), axis=1)
    return pred_ids, pred_proba


class SpanBERT:
    def __init__(self, pretrained_dir, model="spanbert-base-cased"):
        assert os.path.exists(pretrained_dir), "Pre-trained model folder does not exist: {}".format(pretrained_dir)
        self.seed = 42
        self.max_seq_length = 128
        self.batch_size = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        self.fp16 = self.n_gpu > 0
        self._set_seed()
        self.label2id = {label: i for i, label in enumerate(label_list)}
        self.id2label = {i: label for i, label in enumerate(label_list)}
        self.num_labels = len(label_list)    
        #self.tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased", do_lower_case=False)
        self.tokenizer = BertTokenizer.from_pretrained(model, do_lower_case=False)

        print("Loading pre-trained spanBERT from {}".format(pretrained_dir))
        self.classifier = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=self.num_labels)
        if self.fp16:
            self.classifier.half()
        self.classifier.to(self.device)

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)

    def predict(self, examples):
        features = convert_examples_to_features(examples, self.max_seq_length, self.tokenizer, special_tokens)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        dataloader = DataLoader(data, batch_size=self.batch_size)
        preds, proba = predict(self.classifier, self.device, dataloader)
        preds = [self.id2label[pred] for pred in preds]
        return list(zip(preds, proba))

if __name__ == "__main__":
    pretrained_dir = os.path.abspath("./pretrained_spanbert")
    bert = SpanBERT(pretrained_dir=pretrained_dir)
    examples = [
            {"tokens": "Bill Gates is the founder of Microsoft".split(), "subj": ('Bill Gates', "PERSON", (0,1)), "obj": ('Microsoft', "ORGANIZATION", (6,6))},
            {"tokens": "Bill Gates is the founder of Microsoft".split(), "obj": ('Bill Gates', "PERSON", (0,1)), "subj": ('Microsoft', "ORGANIZATION", (6,6))}
            ]
    preds = bert.predict(examples)
    for example, pred in list(zip(examples, preds)):
        example["relation"] = pred
        print(example)

