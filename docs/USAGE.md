# Usage

## Add the source path

```matlab
addpath(genpath('src'));
```

## Load a dataset

```matlab
data = load(fullfile('data', 'Australian.mat'));
X = double(data.X);
Y = double(data.Y);
```

`X` is an `n x d` matrix and `Y` is an `n x 1` vector with labels `+1` and `-1`.

## Train once

```matlab
c = 1;
C = 1;
[V1, V2, VV1, VV2, f, FF1, FF2, kk] = CL1QTSVM(c, C, X, Y);
```

- `c`: loss-term regularization parameter.
- `C`: structural-risk regularization parameter.
- `V1`, `V2`: positive and negative surface parameters.
- `FF1`, `FF2`: objective values of the two subproblems at each iteration.
- `kk`: number of iterations.

## Evaluate a trained model

```matlab
[err, f1] = CL1QTSVMerror(X, Y, V1, V2);
accuracy = 1 - err;
```

## Repeated k-fold cross-validation

```matlab
k = 5;
n = 2;
rng(0);
[t, meanACC, stdACC, meanF1, stdF1] = meanCL1QTSVMtest(X, Y, c, C, k, n);
```

This is the protocol used in the paper. `k` is the number of folds and `n` is the number of repetitions.

## Notes

- Labels must be `+1` and `-1`.
- The code uses double precision; convert integer features with `double(X)`.
- The default demo uses a small number of folds/repetitions for speed. Increase them for paper-level results.
- `kfold_indices` uses MATLAB's global random stream. Call `rng(0)` before the experiment for reproducibility.
