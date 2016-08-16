clear; clc

dim = 2;
mean_true = rand(dim, 1);
cov_true = zeros(dim, dim);
while det(cov_true) < 0.5
    cov = rand(dim, dim);
    cov_true = cov * cov';
end
det(cov_true)
norm_real = log(sqrt((2 * pi)^dim * det(cov_true)));

sample_num = 10000;
data = mvnrnd(mean_true, cov_true, sample_num);
sqrt((2*pi)^dim * det(cov_true))
plot(data(:, 1), data(:, 2), '.'); axis equal

noise_size = 100000;
noise = mvnrnd(zeros(dim, 1), eye(dim, dim), noise_size);

mean = rand(dim , 1);
cov = zeros(dim, dim);
while det(cov) < 0.5
    cov = rand(dim, dim);
    cov = cov * cov';
end
det(cov)
norm_c = rand(1)*10;

%mean = mean_true;
%cov = cov_true;
%norm_c = real_norm;

w = [mean; cov(:); norm_c];
tic;
[loss, grad] = nce_loss(w, data, noise);
toc

%%
f = @(x)nce_loss(x, data, noise);
[loss, grad] = nce_loss(w, data, noise);
%opts = optimoptions(@fminunc,'DerivativeCheck', 'on', ...
%        'Diagnostics', 'on', 'Display', 'iter-detailed', ...
%        'FunValCheck', 'off', 'GradObj', 'on');
opts = optimoptions(@fminunc,'DerivativeCheck', 'off', ...
    'Diagnostics', 'on', 'Display', 'iter-detailed', ...
    'FunValCheck', 'off', 'GradObj', 'on', 'Algorithm', 'quasi-newton', ...
    'MaxIter', length(w) * 100, 'TolFun', 1e-10, 'TolX', 1e-10);
w_res = fminunc(f, w, opts);

mean = w_res(1:dim);
cov = reshape(w_res(dim + 1 : dim + dim ^ 2), dim, dim);
norm_c = w_res(end);

disp('result is:')
fprintf('mean norm(log10) diff: %f\n', log10(norm(mean - mean_true)));
fprintf('cov norm(log10) diff: %f\n', log10(norm(cov - cov_true)));
fprintf('norm const norm(log10) diff: %f\n', log10(abs(norm_c - norm_real)));

