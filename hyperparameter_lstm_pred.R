library(keras)
source("custom_metrics.R")

# Flags -------------------------------------------------------------------

FLAGS <- flags(
  flag_numeric("units", 8),
  flag_numeric("layers", 1),
  flag_integer("N", 5000, "Number of observations for each variable"),
  flag_numeric("val_split", 0.2, "Validation split"),
  flag_numeric("test_split", 0.2, "Test split"),
  flag_integer("lookback", 3, "Input sequence length"),
  flag_integer("batch_size", 128),
  flag_integer("epochs", 20)
)


# Data Preparation --------------------------------------------------------

set.seed(12345)

A <- matrix(rnorm(25),5,5)

# source signals with unit variances
# AR(2)
s1 <- arima.sim(n=FLAGS$N, list(ar=c(0.3639, 0.5941)), sd=0.3563)
# AR(3)
s2 <- arima.sim(n=FLAGS$N, list(ar=c(0.1159,0.3853,0.1756)), sd=0.8449)
# MA(8)
s3 <- arima.sim(n=FLAGS$N, list(ma=c(0.1216, 0.9385, 0.9409, 0.9144, 0.3823, 0.6862, 0.8894, 0.7561)), sd=0.4227)
# AR(5)
s4 <- arima.sim(n=FLAGS$N, list(ar=c(0.199, 0.2152, 0.153, 0.1435, 0.151)), sd=0.7118)
# AR(7)
s5 <- arima.sim(n=FLAGS$N, list(ar=c(0.1452, 0.05336, 0.08183, 0.1739, 0.06431, 0.2536, 0.1341)), sd=0.6991)

S <- cbind(s1, s2, s3, s4, s5)

dataset <- tcrossprod(S, A)

# add artificial means
X <- sweep(dataset, 2, c(1,2,3,4,5), "+")

train_split <- 1 - FLAGS$val_split - FLAGS$test_split
train_size <- train_split * FLAGS$N
val_size <- FLAGS$val_split * FLAGS$N
test_size <- FLAGS$test_split * FLAGS$N

# Normalize data
train_data <- X[1:train_size,]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(X, center = mean, scale = std)

lookback <- FLAGS$lookback
batch_size <- FLAGS$batch_size


val_start <- train_size + 1
val_end <- train_size + val_size

test_start <- train_size + val_size + 1
test_end <- FLAGS$N

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
  data[test_start:test_end,],
  data[test_start:test_end,],
  length = lookback,
  batch_size = FLAGS$batch_size
)

train_steps <- ceiling((train_size - lookback) / batch_size)
val_steps <- ceiling((val_end - val_start - lookback) / batch_size)
test_steps <- ceiling((test_end - test_start - lookback) / batch_size)

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

history <- model %>% fit_generator(
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

model %>% evaluate_generator(test_gen, test_steps)