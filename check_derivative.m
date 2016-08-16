%%

delta = 1e-8;
[loss, grad] = nce_loss(w, data, noise);
grad_numeric = zeros(length(w), 1);
for i = 1:length(w)
    w_p = w;
    w_p(i) = w(i) + delta;
    w_n = w;
    w_n(i) = w(i) - delta;
    [loss_p, ~] = nce_loss(w_p, data, noise);
    [loss_n, ~] = nce_loss(w_n, data, noise);
    grad_numeric(i) = (loss_p - loss_n) / 2 / delta;
end
norm(grad-grad_numeric)


