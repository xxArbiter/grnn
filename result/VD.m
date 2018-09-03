function vd = VD(y, pred)
error = y - pred;
vd = mean(mean((error - mean(mean(error))).^2));