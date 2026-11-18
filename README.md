# Decision Trees with Gradient Descent: A Reproduction of GradTree

## Implementations

- [model/jiayang/](model/jiayang/README.md): GradTree (PyTorch), CART baseline and the paper's
  feature-selection ablation on all five datasets, with setup, commands to reproduce, and results.
- [model/myat/](model/myat/): GradTree (PyTorch), baselines and analysis on all five datasets, over 10 trials (seeds 42-51).
  - [gradtree/](model/myat/gradtree/README.md): the model, training with the paper's tuned and default
    settings, and our own hyperparameter search.
  - [baselines/](model/myat/baselines/README.md): CART (paper's tuned and default settings, our own
    search) and a random forest as an upper reference.
  - [analysis/](model/myat/analysis/README.md): GradTree vs CART with Wilcoxon tests, the depth study, the
    feature-selection ablation (entmax, softmax, Gumbel-softmax, temperature annealing), error analysis,
    tree figures and a leave-one-out encoding leakage check.

## Team Responsibilities

### Data Preprocessing

- Jiayang: Echocardiogram and Iris
- Myat: Spambase and Wine
- Sunny: Adult

### 6.2. Model implementation:

- Jiayang: Baseline CART model
- Myat & Sunny: GradTree model

### 6.3. Model training and experimentation:

- Everyone

### 6.4. Evaluation and error analysis:

- Jiayang: Calculate evaluation metrics
- Sunny: Analyze incorrect predictions
- Myat: Analyze the models' strengths, limitations, and differences across datasets

### 6.5. Report writing and presentation:

- Everyone
