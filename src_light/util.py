import argparse
import albumentations
import os
import numpy as np

import torch

def getargs():
    """
    Get runtime arguments.
    """
    parser = argparse.ArgumentParser()

    #parser.add_argument('--mode', type = str, required = True)
    parser.add_argument('--img-dir', type = str, required = True)
    parser.add_argument('--log-dir', type = str, required = True)
    parser.add_argument('--model-dir', type = str, required = True)
    parser.add_argument('--name', type = str, required = True)

    parser.add_argument('--gpus', type = int, default = 0)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--img-dim', type = int, default = 32)
    parser.add_argument('--batch-size', type = int, default = 48)
    parser.add_argument('--num-workers', type = int, default = 4)
    parser.add_argument('--min-img-per-label', type = int, default = 500) # Minimum number of images per label.
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--epochs', type = int, default = 8)

    parsed_args, _ = parser.parse_known_args()
    return parsed_args

def global_average_precision_score(out_dim, y_true, y_pred, ignore_non_landmarks=False):
    print(out_dim)
    indexes = np.argsort(y_pred[1])[::-1]
    queries_with_target = (y_true < out_dim).sum()
    correct_predictions = 0
    total_score = 0.
    i = 1
    for k in indexes:
        if ignore_non_landmarks and y_true[k] == out_dim:
            continue
        if y_pred[0][k] == out_dim:
            continue
        relevance_of_prediction_i = 0
        if y_true[k] == y_pred[0][k]:
            correct_predictions += 1
            relevance_of_prediction_i = 1
        precision_at_rank_i = correct_predictions / i
        total_score += precision_at_rank_i * relevance_of_prediction_i
        i += 1
    return 1 / queries_with_target * total_score

def append_to_log(args, msg, print_to_console = True):
    with open(os.path.join(args.log_dir, f'{args.name}-log.txt'), 'a') as f:
        f.write(msg + '\n')
    
    if print_to_console:
        print(msg)


def append_to_log(args, msg, print_to_console = True):
    with open(os.path.join(args.log_dir, f'{args.name}-log.txt'), 'a') as f:
        f.write(msg + '\n')
    
    if print_to_console:
        print(msg)