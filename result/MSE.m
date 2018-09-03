function mse = MSE(y, pred)
mse = mean(mean((y-pred).^2));