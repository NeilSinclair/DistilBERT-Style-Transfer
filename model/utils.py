### Utils for DistilBERT-Style-Transfer

# imports
import transformers
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pandas as pd
import numpy as np
import re

import torch.nn.functional as F
import torch.nn

import torch

import datetime



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
        ''' tokenizer - the BART/BERT tokenizer
            create_emb - the create_embedding() method from the BART/BERT model
                         this is only used in a special case of this function
        '''

        self.tokenizer = tokenizer
        self.create_emb = create_emb

    def __call__(self, sample):
        ''' sample is a line from the text with columns [Original, Masked, Label] '''
        sample = encode_single_sentence(self.tokenizer, self.create_emb, sample['Masked'], sample['Original'],
                                        sample['Label'])
        return sample

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

def sentence_rewriter(mask_ids, g_samp, original_sentences, vocab, device='cuda'):
    ''' Function that rewrites sentences with the generated words
      Args: mask_ids - a matrix indicating with a 0 where the masked tokens are in each sentence
            gs_one_hot_mat - matrix of gumbel_softmax predictions; essentially one hot encodings
                             of the sampled token, 1 row for each masked token
            original_sentences - the original sentences
            vocab - a tensor containing np.arange(0, len(tokenizer vocab))
    '''
    mask_idx = mask_ids.nonzero()
    # Add the new words in
    curr_idx = 0
    # store the first masked indices
    mask_idx_temp = mask_idx[0, ].unsqueeze(0)
    g_samp_temp = g_samp[0, ].unsqueeze(0)
    mask_idx_new = []

    # create a nested list of the mask_idx tensors, 1 position for each source sentnece
    for i, idx in enumerate(mask_idx[1:, :]):
        # If the sentence we're processing is the same as the last one, make amendments accordingly
        if idx[0].item() == curr_idx:
            if len(idx.size()) == 1:
                idx_ = idx.unsqueeze(0)
            else:
                idx_ = idx
            mask_idx_temp = torch.cat((mask_idx_temp, idx_), dim=0)
            g_samp_temp = torch.cat((g_samp_temp, g_samp[i].unsqueeze(0)), dim=0)
            # If this is the last sentence of the batch, make the amendments
            if idx[0].item() == mask_idx[-1, 0].item():
                original_sentences[idx[0]] = reconstitute_sentence(g_samp_temp, vocab,
                                                                   original_sentences[idx[0]],
                                                                   mask_ids[idx[0]],
                                                                   mask_idx_temp)

        # If we've moved onto the next sentence, then make the amendments
        else:
            original_sentences[idx[0] - 1] = reconstitute_sentence(g_samp_temp, vocab,
                                                                   original_sentences[idx[0] - 1],
                                                                   mask_ids[idx[0] - 1],
                                                                   mask_idx_temp)
            mask_idx_temp = idx.unsqueeze(0)
            g_samp_temp = g_samp[i].unsqueeze(0)
            curr_idx += 1

    return original_sentences.type(torch.LongTensor)


# Create a timing function
def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))

    return str(datetime.timedelta(seconds=elapsed_rounded))


def reconstitute_sentence(g_samp, voc, original_sent, mask_ids, mask_idx):
    ''' Function that returns a sentence with words substituted by gumbel_softmax sampling
    Args: g_samp - one_hot_encoding sampled using gumbel softmax
          vocab - a tensor of the vocab indexes
          mask_idx - the mask indices, a tensor of [sentence_number, mask_position]
          original_sent - the original sentence (tokens)
          mask_ids - a matrix of 1's and 0's indicating where the masked ids are
    Returns a reconstituted sentence
    '''
    sentence_parts = None
    for i in range(mask_idx.size()[0] + 1):
        if i >= mask_idx.size()[0]:
            # print(f"final sentence piece being added is {(original_sent * (1-mask_ids))}")
            # print(f"mask_ids looks like {mask_ids}")
            # print(f"original_sent looks like {original_sent}")
            sentence_parts = torch.cat((sentence_parts.squeeze(1), (original_sent * (1 - mask_ids)).unsqueeze(0)),
                                       dim=0)
            # print(f"And, final sentence parts are {sentence_parts}")
        else:
            tmp_ = torch.zeros_like(mask_ids)
            if len(mask_idx.size()) == 1:
                tmp_[mask_idx[i]] = 1
                tmp_ = tmp_.unsqueeze(0)
            else:
                tmp_[mask_idx[i, 1]] = 1
            diag = torch.diag(tmp_).type(torch.float32)
            # print(f"g_samp[i]: {g_samp[i].unsqueeze(0)}")
            # print(f"voc: {voc}")
            # print("torch.ones_like(original_sent, dtype = torch.FloatTensor)): {torch.ones_like(original_sent, dtype = torch.FloatTensor))}")

            word = ((g_samp[i].unsqueeze(0) @ voc) * torch.ones_like(original_sent))
            # This creates a tensor of length [sentence] containing just the word that was generated
            # based on what was pulled using the gumbel softmax
            if sentence_parts is None:
                sentence_parts = (word @ diag).unsqueeze(0)
                # print(f"first sentence_part is {sentence_parts}")
            else:
                x_ = (word @ diag).unsqueeze(0)
                # print(f"x_ is {x_}")
                sentence_parts = torch.cat((sentence_parts, x_), dim=0)

    return sentence_parts.sum(dim=0)