mirror <- "https://cloud.r-project.org/"

if(!require(tidyverse)){
  install.packages("tidyverse", repos = mirror)
  library(tidyverse)
}

if(!require(rjson)){
  install.packages("rjson", repos = mirror)
  library(rjson)
}

fetch_measurements <- function(path_to_dir) {
  flags <- fromJSON(file = paste(path_to_dir, "flags.json", sep = "/"))
  evaluation <- fromJSON(file = paste(path_to_dir, "evaluation.json", sep = "/"))
  iteration <- as.integer(str_extract(path_to_dir, "(?<=_)\\d+(?=/)"))
  config_id <- as.integer(str_extract(path_to_dir, "(?<=runs/)\\d+(?=/tfruns)"))
  
  tibble_row(n = flags$N, iteration = iteration, config_id = config_id, 
             units = flags$units, layers = flags$layers, lookback = flags$lookback, 
             steps = flags$steps, batch_size = flags$batch_size,
             mse = evaluation$mean_squared_error, mse_rowsum = evaluation$mse_rowsum, mse_multistep_rowsum = evaluation$mse_multistep_rowsum)
}

list.dirs("simulated_data", recursive = FALSE)

ns <- c(200, 500, 2000, 5000)

get_dataset_dirs <- function(n) {
  list.dirs(paste("simulated_data", n, sep = "/"), recursive = FALSE)
}

dataset_directory_paths <- lapply(ns, get_dataset_dirs) %>% 
  lapply(function(dir_name) paste(dir_name, "runs", sep="/")) %>% 
  lapply(list.files, recursive = FALSE, pattern = "\\d", include.dirs = TRUE, full.names = TRUE) %>%
  lapply(function(dir_name) paste(dir_name, "tfruns.d", sep="/")) %>% 
  unlist()

nn_results <- lapply(dataset_directory_paths, fetch_measurements) %>% 
  do.call(rbind, .)

saveRDS(nn_results, "nn_results.rds")


