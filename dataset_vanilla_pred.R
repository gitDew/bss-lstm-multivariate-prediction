source("SOBI_pred_fxns.R")

args <- commandArgs(TRUE)
n <- as.numeric(args[1]) 

# load dataset directory paths
dataset_directory_paths <- list.dirs(paste("simulated_data", n, sep = "/"), recursive = FALSE)

hs <- c(1, 6)

# number of rows for measurements table space pre-allocation
num_rows <- length(n) * 2000 * length(hs) * 2
n_col <- numeric(num_rows)
i_col <- numeric(num_rows)
h_col <- numeric(num_rows)
k_col <- numeric(num_rows)
bss_col <- character(num_rows)
forecast_col <- character(num_rows)
mse_col <- numeric(num_rows)

c <- 1
message("Starting iteration over dataset directories...")
for (dataset_directory_path in dataset_directory_paths) {
  dataset_path <- list.files(path = dataset_directory_path, pattern = "dataset.csv", full.names = TRUE)
  
  dataset <- as.matrix(read.csv(dataset_path, header = TRUE, row.names = "X"))

  i <- as.numeric(strsplit(strsplit(dataset_directory_path, split = "/")[[1]][3], split = "_")[[1]])[2]
  
  for (step in hs) {
    split_index <- n - step
    
    training_set <- dataset[1:split_index,]
    testing_set <- dataset[-(1:split_index),]
    
    arima_pred <- predict_on_source(training_set, step, forecast_arima)
    ets_pred <- predict_on_source(training_set, step, forecast_ets)
    
    arima_mse <- mean(rowSums((testing_set - arima_pred)**2))
    ets_mse <- mean(rowSums((testing_set - ets_pred)**2))
    
    i_col[c] <- i
    n_col[c] <- n
    h_col[c] <- step
    k_col[c] <- NA
    bss_col[c] <- NA
    forecast_col[c] <- "ARIMA"
    mse_col[c] <- arima_mse
    
    c <- c + 1
    
    i_col[c] <- i
    n_col[c] <- n
    h_col[c] <- step
    k_col[c] <- NA
    bss_col[c] <- NA
    forecast_col[c] <- "ETS"
    mse_col[c] <- ets_mse
    
    c <- c + 1
  }
  message("n = ", n, " i = ", i, " done.")
  
}

measurements <- data.frame(n=n_col, i=i_col, h=h_col, k=k_col, bss=bss_col, forecast=forecast_col, mse=mse_col, stringsAsFactors = FALSE)
saveRDS(measurements, file = paste("simulated_data", n, "vanilla_measurements.rds", sep = "/"))