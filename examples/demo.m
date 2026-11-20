% Minimal CL1QTSVM demo.
% Run from the repository root:
%   addpath(genpath('.'));
%   run(fullfile('examples', 'demo.m'));

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'src')));

data = load(fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'Australian.mat'));
X = double(data.X);
Y = double(data.Y);

c = 1;
C = 1;
k = 5;
n = 2;

rng(0);
[t, meanACC, stdACC, meanF1, stdF1] = meanCL1QTSVMtest(X, Y, c, C, k, n);

fprintf('CL1QTSVM demo on Australian.mat\n');
fprintf('Accuracy: %.4f +/- %.4f\n', meanACC, stdACC);
fprintf('F1:       %.4f +/- %.4f\n', meanF1, stdF1);
fprintf('Time:     %.3f s per run\n', t);
