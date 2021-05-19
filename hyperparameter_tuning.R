library(keras)
library(tfruns)

use_condaenv("tf-gpus", required=TRUE)

flags <- list(
  N = c(5000, 2000, 500, 200),
  val_split = 0.2,
  test_split = 0.2,
  units = c(8, 16, 32, 64, 128, 256, 512),
  lookback = c(3,6,12), # CAREFUL! dataset standardized with max lookback 12 in mind! 
  batch_size = c(32, 64, 128),
  layers = c(1, 2, 3, 4),
  epochs = 1000 # early stopping callback used
)

tuning_run("hyperparameter_lstm_pred.R", flags = flags)

for (n in flags$N) {
  copy_run(ls_runs(subset = flag_N == n, order = "eval_mse_rowsum", decreasing = FALSE)[1:5,], to=paste0("best_runs/", n))
}


# best models can then be rebuilt and retrained by using training_run with the flags.json from the separate run directories.