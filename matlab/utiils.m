function gradJ_diff = backprop_approx(cost_func, Weights, delta)
%%GRADJ_DIFF Approximated backpropogation using forward differencing
% to check implementation.

gradJ_diff = zeros(length(Weights), 1);
for i = 1:length(Weights)
    dWeights = zeros(length(Weights), 1);
    dWeights(i) = delta;

    dJ = cost_func(Weights + dWeights) - cost_func(Weights);
    gradJ_diff(i) = dJ ./ delta;
end