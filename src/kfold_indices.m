function indices = kfold_indices(n, k, seed)
%KFOLD_INDICES Deterministic k-fold partition without toolbox dependencies.
%
%   indices = kfold_indices(n, k) partitions n samples into k folds and
%   returns an n-by-1 vector of fold labels in 1..k.
%
%   indices = kfold_indices(n, k, seed) resets MATLAB's random number
%   generator with the given seed before partitioning. If seed is omitted,
%   the current global random stream is used.

if nargin < 2
    error('kfold_indices:NotEnoughInputs', 'Usage: kfold_indices(n, k, seed).');
end
if k < 2 || k > n
    error('kfold_indices:InvalidK', 'k must satisfy 2 <= k <= n.');
end
if nargin >= 3 && ~isempty(seed)
    rng(seed);
end

perm = randperm(n);
indices = zeros(n, 1);
for fold = 1:k
    indices(perm(fold:k:n)) = fold;
end
end
