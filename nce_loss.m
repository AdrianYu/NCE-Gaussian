function [loss, grad] = nce_loss(w, data, noise)

[sample_size, dim] = size(data);
[noise_size, dim2] = size(noise);
n = length(w);
if dim ^ 2 + dim + 1 ~= n || dim ~= dim2
    loss = 0;
    grad = 0;
    return;
end

mean = w(1:dim);
cov = reshape(w(dim + 1 : dim + dim ^ 2), dim, dim);
norm_c = w(end);
ratio = noise_size / sample_size;
rln = log(ratio);

data_nomean = data - ones(sample_size, 1) * mean';
%u = -0.5 * diag(data_nomean * inv(cov) * data_nomean') - log(norm_c) - log(mvnpdf(data));
u_d = -0.5 * sum((data_nomean / cov).*data_nomean, 2) - norm_c - log(mvnpdf(data));
%r_d = 1. / (1 + ratio * exp(-u_d));
r_d = logsig(u_d - rln);
loss_d = sum(log(r_d));

noise_nomean = noise - ones(noise_size, 1) * mean';
u_n = -0.5 * sum((noise_nomean / cov).*noise_nomean, 2) - norm_c - log(mvnpdf(noise));
%r_n = 1. / (1 + ratio * exp(-u_n));
r_n = logsig(u_n - rln);
loss_n = sum(log(1 - r_n));

loss = -(loss_d + loss_n) / sample_size;

%% 
% gradients
u_d_g = zeros(n, 1);
u_d_ug = cov \ data_nomean';
%u_d_ug = data_nomean / cov;
u_d_cg = -1;
u_d_covg_p = data_nomean / cov;
parfor i = 1:sample_size
    %t_u_d_covg = 0.5 * (cov\data_nomean(i, :)') * (data_nomean(i, :) / cov);
    t_u_d_covg = 0.5 * u_d_covg_p(i, :)' * u_d_covg_p(i, :);
    %t_u_d_covg = u_d_covg_p(i, :)' * u_d_covg_p(i, :);
    %t_u_d_covg = t_u_d_covg - diag(diag(t_u_d_covg)) / 2;
    u_d_g = u_d_g + (1 - r_d(i)) * [u_d_ug(:, i); t_u_d_covg(:); u_d_cg];
end

u_n_g = zeros(n, 1);
u_n_ug = cov \ noise_nomean';
%u_d_ug = data_nomean / cov;
u_n_cg = -1;
u_n_covg_p = noise_nomean / cov;
parfor i = 1:noise_size
    %t_u_n_covg = 0.5 * (cov\noise_nomean(i, :)') * (noise_nomean(i, :) / cov);
    t_u_n_covg = 0.5 * u_n_covg_p(i, :)' * u_n_covg_p(i, :);
    %t_u_n_covg = u_n_covg_p(i, :)' * u_n_covg_p(i, :);
    %t_u_n_covg = t_u_n_covg - diag(diag(t_u_n_covg)) / 2;
    u_n_g = u_n_g - r_n(i) * [u_n_ug(:, i); t_u_n_covg(:); u_n_cg];
end

grad = -(u_d_g + u_n_g) / sample_size;

end


