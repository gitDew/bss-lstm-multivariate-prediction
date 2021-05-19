library(keras)
source("custom_metrics.R")

# Flags -------------------------------------------------------------------

# default values
FLAGS <- flags(
  flag_numeric("units", 8),
  flag_numeric("layers", 1),
  flag_integer("N", 5000, "Number of observations for each variable"),
  flag_numeric("val_split", 0.2, "Validation split"),
  flag_numeric("test_split", 0, "Test split (not used, left in for compatibility reasons)"),
  flag_integer("lookback", 3, "Input sequence length"),
  flag_integer("steps", 6, "multi-step prediction length"),
  flag_integer("batch_size", 128),
  flag_integer("epochs", 20),
  flag_string("dataset_path", "/home/krisz/R/multivariate_lstm_pred/simulated_data/5000/5000_16/5000_16_dataset.csv")
)


# Data Preparation --------------------------------------------------------

data <- as.matrix(read.csv(FLAGS$dataset_path, header = TRUE, row.names = "X"))

# evaluation with the generator for the single step prediction only requires a test set of lookback + 1. The reason why we are reserving
# more here is for the multi-step prediction, which requires a "ground truth" of lookback length, + steps length for targets to compare later.
test_size <- FLAGS$lookback + FLAGS$steps

train_split <- 1 - FLAGS$val_split
train_size <- floor(train_split * (FLAGS$N - test_size))
val_size <- FLAGS$N - train_size - test_size

lookback <- FLAGS$lookback
batch_size <- FLAGS$batch_size

val_start <- train_size + 1
val_end <- train_size + val_size

train_gen <- timeseries_generator(
  data[1:train_size,],
  data[1:train_size,],
  length = lookback,
  shuffle = TRUE,
  batch_size = FLAGS$batch_size
)

val_gen <- timeseries_generator(
  data[val_start:val_end,],
  data[val_start:val_end,],
  length = lookback,
  batch_size = FLAGS$batch_size
)

test_gen <- timeseries_generator(
  tail(data, lookback+1), # NOT test_start:test_end (!!!) since we are only making a single prediction at the end.
  tail(data, lookback+1),
  length = lookback,
  batch_size = FLAGS$batch_size
)

train_steps <- ceiling((train_size - lookback) / batch_size)
val_steps <- ceiling((val_end - val_start - lookback) / batch_size)
test_steps <- 1

features <- dim(data)[[-1]]

# Model Building ----------------------------------------------------------

for (layer in 1:FLAGS$layers) {
  is_not_last_layer <- layer < FLAGS$layers
  if (layer == 1) {
    model <- keras_model_sequential() %>% 
      layer_lstm(units = FLAGS$units, input_shape = list(lookback, features), return_sequences = is_not_last_layer)
  } else {
    model <- model %>% 
      layer_lstm(units = FLAGS$units, return_sequences = is_not_last_layer)
  }
}

model <- model %>%
  layer_dense(units = features)

model %>% compile(
  optimizer = optimizer_adam(),
  loss = metric_mse_rowsum,
  metrics = c("mse", metric_mse_rowsum)
)


# Training ----------------------------------------------------------

model %>% fit_generator(
  train_gen,
  steps_per_epoch = train_steps,
  epochs = FLAGS$epochs,
  validation_data = val_gen,
  validation_steps = val_steps,
  callbacks = list(
    callback_early_stopping(patience = 20, restore_best_weights = TRUE)
  )
)

# Evaluation --------------------------------------------------------------

# single-step evaluation
evaluation_scores <- evaluate_generator(model, test_gen, test_steps)

# multi-step evaluation
test_set <- tail(data, test_size)
seed_data <- head(test_set, lookback)

for (i in 1:FLAGS$steps) {
  input_for_next_prediction <- tail(seed_data, lookback)
  
  # additional dimension (batch size) required for predict function.
  dim(input_for_next_prediction) <- c(1, dim(input_for_next_prediction)[1], dim(input_for_next_prediction)[2])
  
  prediction <- predict(model, input_for_next_prediction, batch_size = 1)
  seed_data <- rbind(seed_data, prediction)
}
final_multi_step_prediction <- tail(seed_data, FLAGS$steps)
multi_step_truth <- tail(test_set, FLAGS$steps)

evaluation_scores$mse_multistep_rowsum <-  mean(rowSums((final_multi_step_prediction - multi_step_truth)**2))
write_run_metadata("evaluation", evaluation_scores)