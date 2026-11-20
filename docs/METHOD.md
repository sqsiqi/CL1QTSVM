# Method

CL1QTSVM is a kernel-free twin support vector machine for binary classification. Instead of using a linear hyperplane or an explicit kernel, it uses a quadratic surface for each class.

## 1. Quadratic feature mapping

For a sample `x`, the code builds a feature vector that contains:

- the linear terms of `x`,
- the upper-triangular quadratic terms,
- a constant term.

This mapping is implemented in `src/X_pro.m`. It lets the model represent curved decision boundaries without a kernel matrix.

## 2. Two quadratic surfaces

The model learns two surfaces:

- a positive-class surface,
- a negative-class surface.

Each surface is parameterized by a symmetric quadratic form, a linear term, and a bias. The two surfaces are optimized alternately: while one surface is updated, the other is fixed.

## 3. Capped L1-norm distance metric

The loss uses a capped L1-norm distance. Large residuals are capped so that noisy samples and outliers cannot dominate the solution. This is the main robustness mechanism of the method.

## 4. Alternating solver

`src/CL1QTSVM.m` performs the alternating updates. It returns:

- the positive and negative surface parameters,
- the objective values of the two subproblems at each iteration,
- the iteration count.

The objective curves can be used to check convergence.

## 5. Prediction

For a test sample, `src/CL1QTSVMerror.m` computes the normalized distance to both surfaces. The sample is assigned to the class with the smaller distance. Accuracy and F1 score are returned.

## 6. Evaluation protocol

`src/meanCL1QTSVMtest.m` runs repeated k-fold cross-validation:

- split the data into `k` folds,
- train on `k-1` folds and test on the held-out fold,
- repeat `n` times,
- report mean and standard deviation of accuracy and F1, plus average runtime.

`src/kfold_indices.m` provides the split without requiring the Bioinformatics Toolbox.
