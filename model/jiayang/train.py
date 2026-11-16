"""Train and evaluate our GradTree over the 10 trials (split seeds 42-51) and compare with the paper.

Run from the repository root:
    python model/jiayang/train.py                                # echocardiogram and iris, seeds 42-51
    python model/jiayang/train.py --datasets iris --seeds 42 43 --jobs 2

Each trial trains on <name>_output/seed_<seed>/ with the paper's tuned hyperparameters (Table 12),
early stopping on the validation split, and reports macro F1 on train / valid / test and the tree
size after pruning. Results go to model/jiayang/results/: runs.csv (one row per trial, saved after
every trial) and summary.csv (mean and std over trials next to the paper's numbers). Trials already
in runs.csv are skipped, so rerunning the same command resumes an interrupted run (--rerun trains them again).
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gradtree import PAPER_HYPERPARAMETERS, GradTree  # noqa: E402

PREPROCESS_DIR = HERE.parents[1] / 'preprocess'
RESULTS_DIR = HERE / 'results'
SEEDS = list(range(42, 52))
MODEL_SEED = 42

# GradTree in the paper, mean and std over 10 trials (as listed in model/myat/config.py):
# test F1 from Tables 1-2, train F1 from Table 8, tree size from Table 9
PAPER_RESULTS = {
    'adult': dict(test_f1=(0.743, 0.034), train_f1=(0.875, 0.050), tree_size=(128.0, 37.19)),
    'echocardiogram': dict(test_f1=(0.658, 0.113), train_f1=(0.827, 0.088), tree_size=(43.2, 12.82)),
    'iris': dict(test_f1=(0.938, 0.057), train_f1=(0.978, 0.017), tree_size=(14.4, 2.54)),
    'spambase': dict(test_f1=(0.903, 0.025), train_f1=(0.905, 0.020), tree_size=(120.2, 37.01)),
    'wine': dict(test_f1=(0.933, 0.031), train_f1=(0.995, 0.009), tree_size=(24.0, 4.12)),
}


def load_split(name, seed):
    folder = PREPROCESS_DIR / f'{name}_output' / f'seed_{seed}'
    if not folder.exists():
        raise FileNotFoundError(f'{folder} is missing: run the preprocessing notebook with SAVE_ALL_SEEDS = True')
    split = {}
    for part in ['train', 'valid', 'test']:
        frame = pd.read_csv(folder / f'{name}_{part}.csv')
        split[f'X_{part}'] = frame.iloc[:, :-1].to_numpy(dtype=np.float32)
        split[f'y_{part}'] = frame.iloc[:, -1].to_numpy(dtype=np.int64)
    split['n_classes'] = int(max(split[f'y_{p}'].max() for p in ['train', 'valid', 'test'])) + 1
    return split


def run_trial(name, seed, epochs, restarts, threads):
    torch.set_num_threads(threads)
    split = load_split(name, seed)
    model = GradTree(split['X_train'].shape[1], split['n_classes'], seed=MODEL_SEED, **PAPER_HYPERPARAMETERS[name])
    start = time.perf_counter()
    model.fit(split['X_train'], split['y_train'], split['X_valid'], split['y_valid'], epochs=epochs, restarts=restarts)
    row = {'dataset': name, 'seed': seed}
    for part in ['train', 'valid', 'test']:
        pred = model.predict(split[f'X_{part}'])
        row[f'f1_{part}'] = f1_score(split[f'y_{part}'], pred, average='macro')
        row[f'accuracy_{part}'] = accuracy_score(split[f'y_{part}'], pred)
    row['tree_size'] = model.tree_size()
    row['best_valid_loss'] = model.best_valid_loss
    row['epochs'] = len(model.history)
    row['fit_seconds'] = time.perf_counter() - start
    return row


def summarize(runs):
    rows = []
    for name, group in runs.groupby('dataset', sort=False):
        paper = PAPER_RESULTS[name]
        rows.append({
            'dataset': name,
            'trials': len(group),
            'test_f1': group['f1_test'].mean(),
            'test_f1_std': group['f1_test'].std(ddof=0),
            'paper_test_f1': paper['test_f1'][0],
            'paper_test_f1_std': paper['test_f1'][1],
            'train_f1': group['f1_train'].mean(),
            'paper_train_f1': paper['train_f1'][0],
            'tree_size': group['tree_size'].mean(),
            'tree_size_std': group['tree_size'].std(ddof=0),
            'paper_tree_size': paper['tree_size'][0],
            'test_accuracy': group['accuracy_test'].mean(),
            'fit_seconds': group['fit_seconds'].mean(),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', default=['echocardiogram', 'iris'], choices=list(PAPER_HYPERPARAMETERS))
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--jobs', type=int, default=4, help='trials trained in parallel')
    parser.add_argument('--epochs', type=int, default=10_000, help='maximum epochs (lower only for quick tests)')
    parser.add_argument('--restarts', type=int, default=3)
    parser.add_argument('--rerun', action='store_true', help='train again trials already in results/runs.csv')
    args = parser.parse_args()

    for name in args.datasets:  # fail before training if a split is missing
        for seed in args.seeds:
            load_split(name, seed)

    # runs.csv is updated after every trial, so an interrupted run resumes with the trials still missing
    RESULTS_DIR.mkdir(exist_ok=True)
    runs_path = RESULTS_DIR / 'runs.csv'
    runs = pd.read_csv(runs_path) if runs_path.exists() else pd.DataFrame(columns=['dataset', 'seed'])
    done = set(zip(runs['dataset'], runs['seed'])) if not args.rerun else set()
    trials = [(name, seed) for name in args.datasets for seed in args.seeds if (name, seed) not in done]
    print(f'{len(trials)} trials to train ({len(args.datasets) * len(args.seeds) - len(trials)} already in {runs_path.name})', flush=True)

    order = list(PAPER_HYPERPARAMETERS)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_trial, name, seed, args.epochs, args.restarts, 1) for name, seed in trials]
        for future in as_completed(futures):
            row = future.result()
            print(f"{time.strftime('%H:%M')} {row['dataset']:15s} seed {row['seed']}: test F1 {row['f1_test']:.3f}, "
                  f"{row['tree_size']} nodes, {row['epochs']} epochs, {row['fit_seconds']:.0f}s", flush=True)
            keep = ~((runs['dataset'] == row['dataset']) & (runs['seed'] == row['seed']))
            runs = pd.concat([runs[keep], pd.DataFrame([row])], ignore_index=True)
            runs = runs.sort_values(['dataset', 'seed'], key=lambda c: c.map(order.index) if c.name == 'dataset' else c)
            runs.to_csv(runs_path, index=False)

    summary = summarize(runs)
    summary.to_csv(RESULTS_DIR / 'summary.csv', index=False)
    with pd.option_context('display.float_format', '{:.3f}'.format, 'display.width', 200):
        print()
        print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
