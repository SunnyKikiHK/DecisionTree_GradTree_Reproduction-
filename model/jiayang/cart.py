"""CART baseline over the 10 trials (split seeds 42-51), compared with GradTree and the paper.

Run from the repository root (after model/jiayang/train.py, whose results it puts next to CART's):
    python model/jiayang/cart.py

As in the paper (Section 4.1, Appendix D): sklearn's DecisionTreeClassifier with the optimized
hyperparameters of Table 15, fitted on the complete training data (our train + valid split, since
CART does not need a validation set for early stopping; the official code does the same).
Writes model/jiayang/results/cart_runs.csv, cart_summary.csv and comparison.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train import MODEL_SEED, PAPER_RESULTS, RESULTS_DIR, SEEDS, load_split  # noqa: E402

DATASETS = ['adult', 'echocardiogram', 'spambase', 'iris', 'wine']

# Table 15 of the paper (CART hyperparameters)
PAPER_CART_HYPERPARAMETERS = {
    'adult': dict(max_depth=10, criterion='entropy', max_features=None, min_samples_leaf=10, min_samples_split=2, ccp_alpha=0.0),
    'echocardiogram': dict(max_depth=9, criterion='entropy', max_features=None, min_samples_leaf=1, min_samples_split=5, ccp_alpha=0.0),
    'spambase': dict(max_depth=10, criterion='gini', max_features=None, min_samples_leaf=1, min_samples_split=2, ccp_alpha=0.0),
    'iris': dict(max_depth=8, criterion='entropy', max_features=None, min_samples_leaf=5, min_samples_split=10, ccp_alpha=0.0),
    'wine': dict(max_depth=9, criterion='gini', max_features=None, min_samples_leaf=1, min_samples_split=5, ccp_alpha=0.0),
}

# CART in the paper, mean and std over 10 trials: test F1 from Tables 1-2, train F1 from Table 8,
# tree size from Table 9
PAPER_CART_RESULTS = {
    'adult': dict(test_f1=(0.771, 0.011), train_f1=(0.973, 0.001), tree_size=(156.6, 23.08)),
    'echocardiogram': dict(test_f1=(0.555, 0.111), train_f1=(0.981, 0.009), tree_size=(36.2, 5.81)),
    'spambase': dict(test_f1=(0.917, 0.011), train_f1=(0.965, 0.003), tree_size=(201.0, 14.64)),
    'iris': dict(test_f1=(0.937, 0.046), train_f1=(0.973, 0.010), tree_size=(9.4, 1.50)),
    'wine': dict(test_f1=(0.907, 0.042), train_f1=(1.000, 0.000), tree_size=(13.4, 2.33)),
}


def run_trial(name, seed):
    split = load_split(name, seed)
    X_fit = np.vstack([split['X_train'], split['X_valid']])
    y_fit = np.concatenate([split['y_train'], split['y_valid']])
    model = DecisionTreeClassifier(random_state=MODEL_SEED, **PAPER_CART_HYPERPARAMETERS[name]).fit(X_fit, y_fit)
    row = {'dataset': name, 'seed': seed}
    for part, X, y in [('train', X_fit, y_fit), ('test', split['X_test'], split['y_test'])]:
        pred = model.predict(X)
        row[f'f1_{part}'] = f1_score(y, pred, average='macro')
        row[f'accuracy_{part}'] = accuracy_score(y, pred)
    row['tree_size'] = model.tree_.node_count  # internal + leaf nodes, as GradTree's tree_size
    row['depth'] = model.get_depth()
    return row


def mean_std(values):
    return f'{np.mean(values):.3f} ± {np.std(values):.3f}'


def main():
    runs = pd.DataFrame([run_trial(name, seed) for name in DATASETS for seed in SEEDS])
    runs.to_csv(RESULTS_DIR / 'cart_runs.csv', index=False)

    summary = pd.DataFrame([{
        'dataset': name,
        'trials': len(group),
        'test_f1': group['f1_test'].mean(),
        'test_f1_std': group['f1_test'].std(ddof=0),
        'paper_test_f1': PAPER_CART_RESULTS[name]['test_f1'][0],
        'paper_test_f1_std': PAPER_CART_RESULTS[name]['test_f1'][1],
        'train_f1': group['f1_train'].mean(),
        'paper_train_f1': PAPER_CART_RESULTS[name]['train_f1'][0],
        'tree_size': group['tree_size'].mean(),
        'tree_size_std': group['tree_size'].std(ddof=0),
        'paper_tree_size': PAPER_CART_RESULTS[name]['tree_size'][0],
        'test_accuracy': group['accuracy_test'].mean(),
    } for name, group in runs.groupby('dataset', sort=False)])
    summary.to_csv(RESULTS_DIR / 'cart_summary.csv', index=False)

    # GradTree vs CART, ours and the paper's, side by side
    gradtree = pd.read_csv(RESULTS_DIR / 'runs.csv')
    rows = []
    for name in DATASETS:
        g = gradtree[gradtree['dataset'] == name]
        c = runs[runs['dataset'] == name]
        pg, pc = PAPER_RESULTS[name], PAPER_CART_RESULTS[name]
        rows.append({
            'dataset': name,
            'gradtree_test_f1': mean_std(g['f1_test']),
            'cart_test_f1': mean_std(c['f1_test']),
            'paper_gradtree_test_f1': f"{pg['test_f1'][0]:.3f} ± {pg['test_f1'][1]:.3f}",
            'paper_cart_test_f1': f"{pc['test_f1'][0]:.3f} ± {pc['test_f1'][1]:.3f}",
            'gradtree_tree_size': f"{g['tree_size'].mean():.1f}",
            'cart_tree_size': f"{c['tree_size'].mean():.1f}",
            'paper_gradtree_tree_size': pg['tree_size'][0],
            'paper_cart_tree_size': pc['tree_size'][0],
            'gradtree_wins_trials': int((g.set_index('seed')['f1_test'] > c.set_index('seed')['f1_test']).sum()),
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS_DIR / 'comparison.csv', index=False)

    with pd.option_context('display.float_format', '{:.3f}'.format, 'display.width', 250):
        print(summary.to_string(index=False))
        print()
        print(comparison.to_string(index=False))


if __name__ == '__main__':
    main()
