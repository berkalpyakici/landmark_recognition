import argparse
import albumentations

from typing import Dict, Tuple, Any

def getargs():
    """
    Get runtime arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type = str, required = True)
    parser.add_argument('--img-dir', type = str, required = True)
    parser.add_argument('--log-dir', type = str, required = True)
    parser.add_argument('--model-dir', type = str, required = True)
    parser.add_argument('--name', type = str, required = True)

    parser.add_argument('--cuda', type = int, default = 1)
    parser.add_argument('--img-dim', type = int, default = 32)
    parser.add_argument('--batch-size', type = int, default = 48)
    parser.add_argument('--num-workers', type = int, default = 4)
    parser.add_argument('--min-img-per-label', type = int, default = 100) # Minimum number of images per label.
    parser.add_argument('--lr', type = float, default = 0.0001)
    parser.add_argument('--epochs', type = int, default = 20)

    parsed_args, _ = parser.parse_known_args()
    return parsed_args

def make_transform_train(hor_dim, ver_dim):
    return albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ImageCompression(quality_lower=99, quality_upper=100),
        albumentations.Resize(hor_dim, ver_dim),
        albumentations.Normalize()
    ])

def make_transform_val(hor_dim, ver_dim):
    return albumentations.Compose([
        albumentations.Resize(hor_dim, ver_dim),
        albumentations.Normalize()
    ])

def global_average_precision_score(
        y_true: Dict[Any, Any],
        y_pred: Dict[Any, Tuple[Any, float]]
) -> float:
    """
    Compute Global Average Precision score (GAP)
    Parameters
    ----------
    y_true : Dict[Any, Any]
        Dictionary with query ids and true ids for query samples
    y_pred : Dict[Any, Tuple[Any, float]]
        Dictionary with query ids and predictions (predicted id, confidence
        level)
    Returns
    -------
    float
        GAP score
    """
    indexes = list(y_pred.keys())
    indexes.sort(
        key=lambda x: -y_pred[x][1],
    )
    queries_with_target = len([i for i in y_true.values() if i is not None])
    correct_predictions = 0
    total_score = 0.
    for i, k in enumerate(indexes, 1):
        relevance_of_prediction_i = 0
        if y_true[k] == y_pred[k][0]:
            correct_predictions += 1
            relevance_of_prediction_i = 1
        precision_at_rank_i = correct_predictions / i
        total_score += precision_at_rank_i * relevance_of_prediction_i

    return 1 / queries_with_target * total_score