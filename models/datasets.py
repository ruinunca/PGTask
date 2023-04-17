import os
import re
import logging
import json

import pandas as pd

from filelock import FileLock
from datasets import Dataset

from datasets import Dataset

from tqdm import tqdm

import torch


logger = logging.getLogger(__name__)


NULL_TOKEN = '<null>'
GEN_TOKEN = '<gen>'
SEP_TOKEN = '<sep>'


SPECIAL_TOKENS_DICT = {    
    'additional_special_tokens': [NULL_TOKEN, GEN_TOKEN, SEP_TOKEN],  
}

def read_json(path):
    data = []
    with open(path, 'r') as f:
        data = json.load(f)
    return data


##############
# Profile Sentences Dataset
##############

def load_profile_sentences_dataset(data_dir, split, lowercase=True, overwrite_cache=False, max_length=1024):
    cached_features_file = os.path.join(
        data_dir,
        "cached_PROFILE_TURNS_{}".format(split),
    )

    # Make sure only the first process in distributed training processes the dataset,
    # and the others will use the cache.
    lock_path = cached_features_file + ".lock"
    with FileLock(lock_path):

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info(f"Loading features from cached file {cached_features_file}")
            dataset = torch.load(cached_features_file)
        else:
            logger.info(f"Creating features from dataset file at {data_dir}")
            raw_data = read_json(os.path.join(data_dir, '%s.json' % split))                    

            dataset = Dataset.from_list(raw_data)
            logger.info("Split (%s) examples: %s" % (split, len(dataset)))
            logger.info("Saving features into cached file %s" % cached_features_file)
            torch.save(dataset, cached_features_file)
    
    return dataset

