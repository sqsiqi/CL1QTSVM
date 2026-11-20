# CL1QTSVM

Official MATLAB implementation of **A Kernel-Free Quadratic Surface Twin Support Vector Machine with Capped L1-Norm Distance Metric for Robust Classification**.

CL1QTSVM is a kernel-free classifier for robust binary classification. It represents each class by a quadratic surface, solves the two surfaces with an alternating iterative procedure, and uses a capped L1-norm distance metric to reduce the influence of outliers. The repository contains the implementation, benchmark data, and reproducible cross-validation scripts used in the paper.

## Highlights

- Kernel-free quadratic surface classifier: no kernel matrix and no kernel hyper-parameter.
- Twin support vector machine formulation: one quadratic surface per class.
- Capped L1-norm distance metric for robustness to noisy samples and outliers.
- Alternating iterative solver with convergence curves for both objective functions.
- Reproducible k-fold cross-validation for accuracy, F1 score, and runtime.

## Method at a glance

For a sample `x`, the code builds a quadratic feature vector containing the linear term, the upper-triangular quadratic term, and a constant. The model then learns two quadratic surfaces, one for the positive class and one for the negative class. A test sample is assigned to the class whose surface gives the smaller normalized distance.

The capped L1-norm metric limits the contribution of large residuals, which makes the classifier less sensitive to outliers. The two surfaces are updated alternately until convergence; the solver returns both the surface parameters and the objective values at each iteration.

See [`docs/METHOD.md`](docs/METHOD.md) for a concise description and [`docs/USAGE.md`](docs/USAGE.md) for the API.

## Repository layout

```text
src/                    MATLAB source files
  CL1QTSVM.m            Train the two quadratic surfaces
  CL1QTSVMcv.m          k-fold split
  CL1QTSVMerror.m       Prediction and error/F1 evaluation
  X_pro.m               Quadratic feature mapping
  meanCL1QTSVMtest.m    Repeated k-fold cross-validation
  calculate_f1_score.m  F1 score helper
  kfold_indices.m       Toolbox-free k-fold partition
data/                   Benchmark datasets in MATLAB .mat format
examples/demo.m         Minimal runnable example
docs/METHOD.md          Method summary
docs/USAGE.md           Quick-start and API notes
```

The original `代码/` and `数据集/` folders are now `src/` and `data/`. File names and MATLAB function names are unchanged.

## Requirements

- MATLAB R2020a or later.
- No additional toolbox is required. `crossvalind` from the Bioinformatics Toolbox is replaced by the included `kfold_indices` function.

## Quick start

```matlab
addpath(genpath('src'));

data = load(fullfile('data', 'Australian.mat'));
X = double(data.X);
Y = double(data.Y);

c = 1;          % loss-term regularization
C = 1;          % structural-risk regularization
k = 5;          % folds
n = 2;          % repetitions

rng(0);
[t, meanACC, stdACC, meanF1, stdF1] = meanCL1QTSVMtest(X, Y, c, C, k, n);

fprintf('Accuracy: %.4f +/- %.4f\n', meanACC, stdACC);
fprintf('F1:       %.4f +/- %.4f\n', meanF1, stdF1);
fprintf('Time:     %.3f s per run\n', t);
```

Run the same example from the repository root:

```matlab
addpath(genpath('.'));
run(fullfile('examples', 'demo.m'));
```

## Data format

Each `.mat` file contains:

- `X`: an `n x d` feature matrix.
- `Y`: an `n x 1` label vector with values `+1` and `-1`.

Some files also contain a combined matrix (for example, `australian`) kept from the original data release. Files with suffixes such as `5`, `10`, or `15` are the resampled variants used in the paper's experiments.

## Output

`meanCL1QTSVMtest` returns:

- `t`: average runtime per repetition.
- `meanACC`, `std_ACC`: mean and standard deviation of accuracy.
- `meanF1`, `std_F1`: mean and standard deviation of the F1 score.

`CL1QTSVM` also returns the two surface parameter vectors, the objective values of the two subproblems at each iteration, and the iteration count.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{si2025cl1qtsvm,
  title   = {A Kernel-Free Quadratic Surface Twin Support Vector Machine with Capped L1-Norm Distance Metric for Robust Classification},
  author  = {Si, Qi and Yang, Zhi-Xia and others},
  journal = {Manuscript},
  year    = {2025}
}
```

Please also cite the original sources of the benchmark datasets when you use them.

## License

This repository is released under the Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
