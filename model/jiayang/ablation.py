"""Ablation of the split feature selection (paper Section 4.2, Table 5): straight-through entmax
(GradTree) vs straight-through Gumbel-softmax vs temperature annealing, with the optimized
hyperparameters of Table 12.

Run from the repository root, after training each variant:
    python model/jiayang/train.py --datasets echocardiogram iris wine
    python model/jiayang/train.py --datasets echocardiogram iris wine --index-activation gumbel
    python model/jiayang/train.py --datasets echocardiogram iris wine --index-activation annealing
    python model/jiayang/ablation.py

Writes model/jiayang/results/ablation.csv (test macro F1 and tree size, ours next to Table 5).
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train import RESULTS_DIR  # noqa: E402

# Table 5 of the paper (optimized hyperparameters): test macro F1, mean and std over 10 trials
PAPER_ABLATION = {
    'adult': dict(entmax=(0.743, 0.034), gumbel=(0.583, 0.029), annealing=(0.737, 0.045)),
    'echocardiogram': dict(entmax=(0.658, 0.113), gumbel=(0.574, 0.108), annealing=(0.619, 0.151)),
    'spambase': dict(entmax=(0.903, 0.025), gumbel=(0.541, 0.093), annealing=(0.875, 0.019)),
    'iris': dict(entmax=(0.938, 0.057), gumbel=(0.743, 0.224), annealing=(0.930, 0.048)),
    'wine': dict(entmax=(0.933, 0.031), gumbel=(0.565, 0.195), annealing=(0.881, 0.043)),
}
VARIANTS = {'entmax': 'runs.csv', 'gumbel': 'runs_gumbel.csv', 'annealing': 'runs_annealing.csv'}


def main():
    runs = {v: pd.read_csv(RESULTS_DIR / f) for v, f in VARIANTS.items() if (RESULTS_DIR / f).exists()}
    datasets = [d for d in PAPER_ABLATION if all(d in set(r['dataset']) for r in runs.values())]
    rows = []
    for name in datasets:
        row = {'dataset': name}
        for variant, r in runs.items():
            group = r[r['dataset'] == name]
            paper = PAPER_ABLATION[name][variant]
            row[f'{variant}_test_f1'] = f"{group['f1_test'].mean():.3f} ± {group['f1_test'].std(ddof=0):.3f}"
            row[f'paper_{variant}_test_f1'] = f'{paper[0]:.3f} ± {paper[1]:.3f}'
            row[f'{variant}_tree_size'] = round(group['tree_size'].mean(), 1)
            row[f'{variant}_trials'] = len(group)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / 'ablation.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', None):
        print(table.to_string(index=False))


if __name__ == '__main__':
    main()
