import tqdm
import torch
import numpy as np

from .rank import get_rank
from .likelihood import get_ll
from .entropy import get_entropy
from .supervised import eval_supervised
from .metric import get_roc_metrics, get_precision_recall_metrics

def run_baseline_threshold_experiment(args, data, criterion_fn, name, n_samples=500):
    torch.manual_seed(0)
    np.random.seed(0)

    batch_size = args.batch_size

    results = []
    for batch in tqdm.tqdm(range(n_samples // batch_size), desc=f"Computing {name} criterion"):
        original_text = data["original"][batch * batch_size:(batch + 1) * batch_size]
        sampled_text = data["sampled"][batch * batch_size:(batch + 1) * batch_size]

        for idx in range(len(original_text)):
            results.append({
                "original": original_text[idx],
                "original_crit": criterion_fn(original_text[idx]),
                "sampled": sampled_text[idx],
                "sampled_crit": criterion_fn(sampled_text[idx]),
            })

    # compute prediction scores for real/sampled passages
    predictions = {
        'real': [x["original_crit"] for x in results],
        'samples': [x["sampled_crit"] for x in results],
    }

    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"{name}_threshold ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    return {
        'name': f'{name}_threshold',
        'predictions': predictions,
        'info': {
            'n_samples': n_samples,
        },
        'raw_results': results,
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }


def run_baselines(args, config, data):
    n_samples = args.n_samples

    liklihood_criterion = lambda text: get_ll(args, config, text)
    baseline_outputs = [run_baseline_threshold_experiment(args, data, liklihood_criterion, "likelihood", n_samples=n_samples)]

    if args.openai_model is None:
        rank_criterion = lambda text: -get_rank(args, config, text, log=False)
        baseline_outputs.append(run_baseline_threshold_experiment(args, data, rank_criterion, "rank", n_samples=n_samples))
        logrank_criterion = lambda text: -get_rank(args, config, text, log=True)
        baseline_outputs.append(run_baseline_threshold_experiment(args, data, logrank_criterion, "log_rank", n_samples=n_samples))
        entropy_criterion = lambda text: get_entropy(args, config, text)
        baseline_outputs.append(run_baseline_threshold_experiment(args, data, entropy_criterion, "entropy", n_samples=n_samples))

    baseline_outputs.append(eval_supervised(args, data, model='roberta-base-openai-detector'))
    baseline_outputs.append(eval_supervised(args, data, model='roberta-large-openai-detector'))

    return baseline_outputs
