### Utils for DistilBERT-Style-Transfer

# imports
import transformers
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pandas as pd
import numpy as np
import re

import torch.nn.functional as F
import torch.nn
# import pytorch_lightning as pl
import torch
# from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import os

from datetime import timedelta
import datetime
import time


class TokenizedDataset(Dataset):
    ''' class for loading and transforming data into embeddings '''

    def __init__(self, data_file, transform=None, translate=False):
        '''
        Args: data_file - the .csv containing the data_file
              transform - an instantiated transformation class
              translate - whether to encode a translation (i.e. swap the label)
        '''
        self.data = pd.read_csv(data_file)
        self.transform = transform
        self.translate = translate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Get a data item and transform it
        sample = self.data.iloc[idx, :]

        # If we're translating, swap the label from 0 to 1 or 1 to 0
        if self.translate:
            sample[-1] = 1 - sample[-1]

        if self.transform:
            sample = self.transform(sample)
        return sample

    ''' Code taken almost verbatim from utils.py in the transformers/seq2seq github '''

    def collate_fn(self, batch):
        input_ids = torch.stack([x["input_ids"] for x in batch]).squeeze()
        masks = torch.stack([x["attention_mask"] for x in batch]).squeeze()
        target_ids = torch.stack([x["labels"] for x in batch]).squeeze()
        masked_ids = torch.stack([x["masked_ids"] for x in batch]).squeeze()
        class_labels = torch.stack([x["class_label"] for x in batch]).squeeze()

        batch = {
            "input_ids": input_ids,
            "attention_mask": masks,
            "labels": target_ids,
            "masked_ids": masked_ids,
            "class_labels": class_labels
        }
        return batch


class CreateTokens(object):
    ''' Create the embeddings for the sentences passed into the model '''

    def __init__(self, tokenizer, create_emb):
        ''' tokenizer - the BART tokenizer
            create_emb - the create_embedding() method from the BART model '''

        self.tokenizer = tokenizer
        self.create_emb = create_emb

    def __call__(self, sample):
        ''' sample is a line from the text with columns [Original, Masked, Label] '''
        sample = encode_single_sentence(self.tokenizer, self.create_emb, sample['Masked'], sample['Original'],
                                        sample['Label'])
        return sample


encode_single_sentence(self.tokenizer, self.create_emb, sample['Masked'], sample['Original'], sample['Label'])


def encode_single_sentence(tokenizer, model_emb, source_sentence, target_sentence, class_label, max_length=32,
                           pad_to_max_length=True, return_tensors="pt",
                           add_special_tokens=False, return_targets=True):
    ''' Function that tokenizes a sentence
        Args: tokenizer - the BART/BERT tokenizer
              model_emb - function to create embedding; used for other versions of the research
              source_sentence - the source (masked) sentence
              target_sentence - the target (unmasked) sentence
              class_label - the class / style classification of the example
              max_length, pad_to_max_length, return_tensors - tokenizer arguments
              return_targets - whether to return the tokenized targets;
                    not necessary if we're just looking at the validation sentences

        Returns: Dictionary with keys: input_ids, attention_mask, target_ids
    '''

    input_ids = []
    attention_masks = []
    target_ids = []

    sentence = source_sentence
    # Remove unecessary tokens
    sentence = re.sub(r'<s> |</s> ', '', sentence)
    sentence = re.sub(r' {2,10}', ' ', sentence)
    encoded_dict = tokenizer(
        sentence,
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
        add_special_tokens=add_special_tokens
    )

    input_ids.append(encoded_dict['input_ids'])

    attention_masks.append(encoded_dict['attention_mask'])
    # Get the masked ids and then return the location of these masked ids
    masked_ids = torch.where(encoded_dict['input_ids'] == tokenizer.mask_token_id,
                             torch.ones_like(encoded_dict['input_ids']),
                             torch.zeros_like(torch.LongTensor([1])))

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    class_label = torch.LongTensor([class_label])

    # only do this process if we're returning targets (for training)
    if return_targets:
        sentence = target_sentence
        # Remove the <neg> and <pos> tags from the target
        sentence = re.sub(r'<neg> |<pos> ', '', sentence)
        sentence = re.sub(r'<s> |</s> ', '', sentence)
        sentence = re.sub(r' {2,10}', ' ', sentence)
        encoded_dict = tokenizer(
            sentence,
            max_length=max_length,
            padding="max_length" if pad_to_max_length else None,
            truncation=True,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens
        )
        # Shift the target ids to the right
        # shifted_target_ids = shift_tokens_right(encoded_dict['input_ids'], tokenizer.pad_token_id)
        target_ids.append(encoded_dict['input_ids'])

        target_ids = torch.cat(target_ids, dim=0)
    else:
        target_ids = 0

    # If there aren't any masked tokens, then don't return anything (translation wouldn't be possible anyway)
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": target_ids,
        "masked_ids": masked_ids,
        "class_label": class_label
    }

    return batch

def freeze_params(model, action = "freeze"):
  ''' Function that takes a model as input (or part of a model) and freezes the layers for faster training
      adapted from finetune.py '''
  for layer in model.parameters():
    if action == "freeze":
      layer.requires_grad = False
    else:
      layer.requires_grad = True

def freeze_multiple_params(model_method, params):
    '''
    :param model_method: list containing the model methods (e.g. model.distilbert.embeddings.word_embeddings)
                         which should be frozen or not
    :param params: list of hparams indicating which of the above should be frozen or not
    '''
  for i, param in enumerate(params):
    if param:
      freeze_params(model_method[i])
    elif freeze_token_embeds == False:
      freeze_params(model_method[i], action = "no_freeze")
