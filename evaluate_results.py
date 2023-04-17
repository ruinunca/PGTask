import argparse
import os
import re
import json
import random

import csv

from collections import Counter
import evaluate

from nltk.tokenize import word_tokenize


import numpy as np

from tqdm import tqdm
from copy import deepcopy
import torch

def distinct(seqs):
    """ Calculate intra/inter distinct 1/2. """
    batch_size = len(seqs)
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    return intra_dist1, intra_dist2, inter_dist1, inter_dist2


def read_json(path):
    data = []
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def process(args):
    if not os.path.exists(os.path.join(args.experiment, "generated_results.json")):
        raise ValueError("You need to first generate the results using generate_results.py")

    results = read_json(os.path.join(args.experiment, "generated_results.json"))

    bleu_results = []
    bertscore_results = []
    rouge_results = []
    
    bleu = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")
    rouge = evaluate.load("rouge")

    result_sequences = []

    references = []
    bleu_references = []
    predictions = []

    for result in tqdm(results[:5]):
        generated_sentences = result['generated_sentences']
        profile_sentences = " ".join(result['profile_sentences'])

        references.append(profile_sentences)
        bleu_references.append([profile_sentences])
        predictions.append(generated_sentences)
       
        """# BLEU
        bleu_metric = bleu.compute(predictions=[generated_sentences], references=[[profile_sentences]])
        bleu_results.append(bleu_metric['precisions'])

        # BERTScore
        bertscore_metric = bertscore.compute(predictions=[generated_sentences], references=[profile_sentences], lang="en")
        bertscore_results.append(bertscore_metric)

        rouge_metric = rouge.compute(predictions=[generated_sentences], references=[profile_sentences])
        rouge_results.append(rouge_metric)
        # DISTINCT
        #words = word_tokenize(generated_sentences)
        #result_sequences.append(words)

    bleurt_precision = np.array([s['precision'] for s in bertscore_results]).sum(axis=0) / len(bertscore_results)
    bleurt_recall = np.array([s['recall'] for s in bertscore_results]).sum(axis=0) / len(bertscore_results)
    bleurt_f1 = np.array([s['f1'] for s in bertscore_results]).sum(axis=0) / len(bertscore_results)

    bleu = np.array(bleu_results).sum(axis=0) / len(bleu_results)

    rouge_1 = np.array([s['rouge1'] for s in rouge_results]).sum(axis=0) / len(rouge_results)
    rouge_2 = np.array([s['rouge2'] for s in rouge_results]).sum(axis=0) / len(rouge_results)
    rouge_l = np.array([s['rouge2'] for s in rouge_results]).sum(axis=0) / len(rouge_results)


    #intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(result_sequences)
    scores = {
        'bleu': bleu_results,
        'bertscore': {
            'precision': bleurt_precision,
            'recall': bleurt_recall,
            'f1': bleurt_f1,
        },
        'rouge': {
            'rouge1': rouge_1,
            'rouge2': rouge_2,
            'rougel': rouge_l,
        },
       # 'distinct_n1': intra_dist1,
       # 'distinct_n2': intra_dist2,
    }
    """
    bleu_score = bleu.compute(predictions=predictions, references=bleu_references)
    print("BLEU:", bleu_score)
    rouge_score = rouge.compute(predictions=predictions, references=references)
    print("ROUGE:", rouge_score)
    bert_score = bertscore.compute(predictions=predictions, references=references, lang="en")
    precision = np.array(bert_score['precision']).mean().item()
    recall = np.array(bert_score['recall']).mean().item()
    f1 = np.array(bert_score['f1']).mean().item()

    bert_score_complete = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

    scores = {
        'bleu': bleu_score,
        'bertscore': bert_score_complete,
        'rouge': rouge_score,
       # 'distinct_n1': intra_dist1,
       # 'distinct_n2': intra_dist2,
    }

    print(scores)

    output_file = os.path.join(args.experiment, "generation_scores.json")
    with open(output_file, "w",  encoding='utf-8') as writer:
        json.dump(scores, writer, ensure_ascii=False, indent=4)

    config = read_json(os.path.join(args.experiment, "config.json"))


    output_file = os.path.join(args.experiment, "generation_scores.csv")
    row_info = [config['_name_or_path']]
    row = bleu_score['precisions'] + [rouge_score['rouge1'], rouge_score['rouge2'], rouge_score['rougeL']] + [precision, recall, f1]
    with open(output_file, 'w', encoding='UTF8') as f:
        writer = csv.writer(f, delimiter=';')

        # write the data
        writer.writerow(row_info)
        writer.writerow(row)
    



if __name__ == "__main__":
    # ------------------------
    # ARGUMENTS
    # ------------------------
    random.seed(23)

    parser = argparse.ArgumentParser(
        description="Process Profile Sentences",
        add_help=True,
    )
    parser.add_argument(
        "--experiment", 
        type=str, 
        required=True,
        help="Path to experiment with results."
    )

    args = parser.parse_args()

    process(args)