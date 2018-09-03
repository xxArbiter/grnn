function [mse, vd] = metric(prediction, data)
prediction = squeeze(double(prediction));
pred = prediction(size(prediction, 1) - 720 + 1:size(prediction, 1), :);
data = squeeze(data);
y = data(size(data, 1) - 720 + 1:size(data, 1), :)/100;
mse = MSE(y, pred)*10000;
vd = VD(y, pred)*10000;