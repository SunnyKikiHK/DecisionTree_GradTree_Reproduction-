"""GradTree: axis-aligned decision trees learned with gradient descent (PyTorch).

Implements Marton et al., "GradTree: Learning Axis-Aligned Decision Trees with
Gradient Descent" (AAAI 2024):
- dense representation: feature-index logits I and thresholds T (2^d - 1 x n), leaf logits L (2^d x c)
- straight-through entmax for the split feature, straight-through rounding for hard splits (Algorithm 1)
- Glorot uniform initialization (Algorithm 2), Adam per parameter group, mini-batches
- weight averaging over 5 consecutive checkpoints, early stopping on the validation loss, random restarts
- post-hoc pruning of all branches that no training sample reaches
Where the paper is silent, values follow the official experiment code (batch size 128, 10,000 epochs).
"""

from collections import deque

import numpy as np
import torch
import torch.nn.functional as F


# Table 12 of the paper (optimized hyperparameters) for the datasets of this project
PAPER_HYPERPARAMETERS = {
    'echocardiogram': dict(depth=8, lr_index=0.010, lr_values=0.050, lr_leaf=0.100, loss='crossentropy', poly_loss=True, poly_epsilon=5),
    'adult': dict(depth=8, lr_index=0.050, lr_values=0.005, lr_leaf=0.050, loss='crossentropy', poly_loss=True, poly_epsilon=5),
    'spambase': dict(depth=10, lr_index=0.005, lr_values=0.010, lr_leaf=0.010, loss='crossentropy', poly_loss=False, poly_epsilon=2),
    'iris': dict(depth=7, lr_index=0.005, lr_values=0.005, lr_leaf=0.050, loss='crossentropy', poly_loss=False, poly_epsilon=2),
    'wine': dict(depth=10, lr_index=0.010, lr_values=0.050, lr_leaf=0.010, loss='focal', poly_loss=False, poly_epsilon=2),
}


class Entmax15(torch.autograd.Function):
    """1.5-entmax over the last dimension (Peters, Niculae and Martins 2019), exact sort-based solution."""

    @staticmethod
    def forward(ctx, z):
        z = z / 2
        z = z - z.max(dim=-1, keepdim=True).values
        z_sorted = z.sort(dim=-1, descending=True).values
        k = torch.arange(1, z.shape[-1] + 1, dtype=z.dtype, device=z.device)
        mean = z_sorted.cumsum(-1) / k
        mean_sq = (z_sorted ** 2).cumsum(-1) / k
        delta = (1 - k * (mean_sq - mean ** 2)) / k
        tau = mean - delta.clamp(min=0).sqrt()
        support = (tau <= z_sorted).sum(dim=-1, keepdim=True)
        tau_star = tau.gather(-1, support - 1)
        root = (z - tau_star).clamp(min=0)
        ctx.save_for_backward(root)
        return root ** 2

    @staticmethod
    def backward(ctx, upstream):
        # Jacobian of 1.5-entmax: d(root^2)/dz = 2 * root * 1/2 on the support
        root, = ctx.saved_tensors
        g = upstream * root
        return g - g.sum(-1, keepdim=True) / root.sum(-1, keepdim=True) * root


def entmax15(z):
    return Entmax15.apply(z)


def straight_through(soft, hard):
    """Forward pass uses `hard`, backward pass uses the gradient of `soft`."""
    return soft - (soft - hard).detach()


def tree_routing(depth):
    """Internal node index (Equation 4) and path position (Equation 5) for every leaf l and depth j."""
    leaves = np.arange(2 ** depth)[:, None]
    j = np.arange(1, depth + 1)[None, :]
    node_index = 2 ** (j - 1) + leaves // 2 ** (depth - (j - 1)) - 1
    path = (leaves // 2 ** (depth - j)) % 2  # 0 = left branch, 1 = right branch
    return node_index, path


def glorot_uniform(rng, shape):
    limit = np.sqrt(6 / (shape[0] + shape[1]))
    return torch.tensor(rng.uniform(-limit, limit, size=shape), dtype=torch.float32, requires_grad=True)


class GradTree:
    def __init__(self, n_features, n_classes, depth=6, lr_index=0.01, lr_values=0.01, lr_leaf=0.01,
                 loss='crossentropy', focal_gamma=3, poly_loss=False, poly_epsilon=2,
                 average_checkpoints=5, seed=42, feature_names=None):
        self.n_features = n_features
        self.n_classes = n_classes
        self.depth = depth
        self.learning_rates = (lr_index, lr_values, lr_leaf)
        self.loss_name = loss
        self.focal_gamma = focal_gamma
        self.poly_loss = poly_loss
        self.poly_epsilon = poly_epsilon
        self.average_checkpoints = average_checkpoints
        self.seed = seed
        self.feature_names = feature_names or [f'x{i}' for i in range(n_features)]

        self.n_internal = 2 ** depth - 1
        self.n_leaves = 2 ** depth
        node_index, path = tree_routing(depth)
        self.node_index = torch.tensor(node_index, dtype=torch.long)
        self.path = torch.tensor(path, dtype=torch.float32)
        self.history = []

    # ----- model -----

    def init_params(self, seed):
        rng = np.random.RandomState(seed)
        return [
            glorot_uniform(rng, (self.n_internal, self.n_features)),  # I: split feature logits
            glorot_uniform(rng, (self.n_internal, self.n_features)),  # T: split thresholds
            glorot_uniform(rng, (self.n_leaves, self.n_classes)),  # L: leaf class logits
        ]

    def forward(self, X, params):
        """Tree pass function (Algorithm 1) for a whole batch; returns class logits."""
        index_logits, split_values, leaf_logits = params
        soft_index = entmax15(index_logits)
        index = straight_through(soft_index, F.one_hot(soft_index.argmax(-1), self.n_features).float())

        # Equation 6 for every internal node: sigmoid(sum_i iota_i tau_i - sum_i iota_i x_i)
        thresholds = (index * split_values).sum(-1)
        selected_x = X @ index.T
        s = torch.sigmoid(thresholds - selected_x)
        s = straight_through(s, torch.round(s))  # Equation 7, hard split

        # Equation 3: product over the nodes on the path to each leaf. Built level by level: a node's
        # left child gets the node's product times s, its right child times (1 - s). After d levels
        # this is the product along every root-to-leaf path given by Equations 4 and 5, with the same
        # values and gradients, but shared path prefixes are multiplied once instead of once per leaf.
        leaf_membership = torch.ones(X.shape[0], 1, dtype=s.dtype)
        for level in range(self.depth):
            s_level = s[:, 2 ** level - 1:2 ** (level + 1) - 1]  # nodes of this level, left to right
            leaf_membership = torch.stack([leaf_membership * s_level, leaf_membership * (1 - s_level)], dim=2).flatten(1)
        return leaf_membership @ leaf_logits  # Equation 1

    def loss(self, y, logits):
        log_probs = F.log_softmax(logits, dim=-1)
        log_p_true = log_probs.gather(1, y[:, None]).squeeze(1)
        p_true = log_p_true.exp()
        loss = -log_p_true
        if self.loss_name == 'focal':
            loss = (1 - p_true) ** self.focal_gamma * loss  # focal factor (Lin et al. 2017)
        if self.poly_loss:
            loss = loss + self.poly_epsilon * (1 - p_true)  # Poly-1 (Leng et al. 2022)
        return loss.mean()

    # ----- training (Algorithm 2) -----

    def fit(self, X_train, y_train, X_valid, y_valid, batch_size=128, epochs=10_000, patience=200,
            restarts=3, verbose=False):
        X_train = torch.tensor(np.asarray(X_train), dtype=torch.float32)
        y_train = torch.tensor(np.asarray(y_train), dtype=torch.long)
        X_valid = torch.tensor(np.asarray(X_valid), dtype=torch.float32)
        y_valid = torch.tensor(np.asarray(y_valid), dtype=torch.long)
        n = len(y_train)
        batch_size = min(batch_size, int(np.ceil(n / 2)))

        best_loss, best_params = np.inf, None
        self.history = []
        for restart in range(restarts + 1):
            seed = self.seed + restart
            params = self.init_params(seed)
            # One Adam per parameter group, each with its own learning rate (eps as in TensorFlow)
            optimizer = torch.optim.Adam([{'params': [p], 'lr': lr} for p, lr in zip(params, self.learning_rates)], eps=1e-7)

            checkpoints = deque(maxlen=self.average_checkpoints)
            run_best_loss, run_best_params, wait = np.inf, None, 0
            rng = np.random.RandomState(seed)
            for epoch in range(epochs):
                order = torch.tensor(rng.permutation(n))
                for start in range(0, n, batch_size):
                    batch = order[start:start + batch_size]
                    optimizer.zero_grad()
                    self.loss(y_train[batch], self.forward(X_train[batch], params)).backward()
                    optimizer.step()

                # Weight averaging: evaluate the mean of the last checkpoints (one per epoch)
                checkpoints.append([p.detach().clone() for p in params])
                averaged = [torch.stack(c).mean(0) for c in zip(*checkpoints)]
                with torch.no_grad():
                    current = self.loss(y_valid, self.forward(X_valid, averaged)).item()
                self.history.append((restart, epoch, current))

                if current < run_best_loss:
                    run_best_loss, run_best_params, wait = current, averaged, 0
                else:
                    wait += 1
                    if wait >= patience:
                        break

            if verbose:
                print(f'restart {restart}: best valid loss {run_best_loss:.4f} after {epoch + 1} epochs')
            if run_best_loss < best_loss:
                best_loss, best_params = run_best_loss, run_best_params

        self.best_valid_loss = best_loss
        self.params = best_params
        self.prune(X_train.numpy())
        return self

    # ----- hard tree, pruning and inference -----

    def prune(self, X_train):
        """Convert to a vanilla tree and remove all branches that no training sample reaches."""
        index_logits, split_values, leaf_logits = [p.detach().numpy() for p in self.params]
        features = np.argmax(index_logits, axis=1)
        thresholds = split_values[np.arange(self.n_internal), features]
        leaf_probs = torch.softmax(torch.tensor(leaf_logits), dim=-1).numpy()

        def build(node, X):
            if node >= self.n_internal:
                leaf = node - self.n_internal
                return {'leaf': leaf, 'probs': leaf_probs[leaf], 'samples': len(X)}
            go_left = X[:, features[node]] < thresholds[node]  # S = 1 -> left branch
            left = build(2 * node + 1, X[go_left])
            right = build(2 * node + 2, X[~go_left])
            if left['samples'] == 0:
                return right
            if right['samples'] == 0:
                return left
            return {'feature': int(features[node]), 'threshold': float(thresholds[node]),
                    'left': left, 'right': right, 'samples': len(X)}

        self.tree = build(0, np.asarray(X_train, dtype=np.float32))
        return self.tree

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        probs = np.zeros((len(X), self.n_classes), dtype=np.float32)

        def route(node, rows):
            if 'leaf' in node:
                probs[rows] = node['probs']
                return
            go_left = X[rows, node['feature']] < node['threshold']
            route(node['left'], rows[go_left])
            route(node['right'], rows[~go_left])

        route(self.tree, np.arange(len(X)))
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def tree_size(self):
        """Number of internal and leaf nodes after pruning."""
        def count(node):
            return 1 if 'leaf' in node else 1 + count(node['left']) + count(node['right'])
        return count(self.tree)

    def export_text(self, class_names=None):
        lines = []

        def write(node, indent):
            pad = '|   ' * indent
            if 'leaf' in node:
                label = int(np.argmax(node['probs']))
                name = class_names[label] if class_names is not None else label
                lines.append(f'{pad}class: {name} ({node["samples"]} train samples)')
                return
            name = self.feature_names[node['feature']]
            lines.append(f'{pad}{name} < {node["threshold"]:.4f}')
            write(node['left'], indent + 1)
            lines.append(f'{pad}{name} >= {node["threshold"]:.4f}')
            write(node['right'], indent + 1)

        write(self.tree, 0)
        return '\n'.join(lines)
