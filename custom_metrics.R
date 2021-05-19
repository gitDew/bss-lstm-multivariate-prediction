# custom metric for tensorflow/keras to calculate the mean squared error by summing up the rows of the squared errors.
# this metric is required for direct comparison on the same scale between the neural network performances and the blind source separation method performances.
metric_mse_rowsum <- custom_metric("mse_rowsum", function(y_true, y_pred) {
  k_mean(k_sum(k_square(y_true - y_pred), axis = 2), axis = 1)
})