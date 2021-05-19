mirror <- "https://cloud.r-project.org/"
if(!require(keras)){
  install.packages("keras", repos = mirror)
  library(keras)
}

if(!require(yaml)){
  install.packages("yaml", repos = mirror)
  library(yaml)
}

if(!require(tfruns)){
  install.packages("tfruns", repos = mirror)
  library(tfruns)
}

args <- commandArgs(TRUE)
n <- as.numeric(args[1]) 

if (args[2] == "GPU") {
  use_condaenv("tf-gpus", required=TRUE)
} else if (args[2] == "CPU") {
  use_condaenv("tf-cpus", required=TRUE)
}

# load flags
best_run_flags <- list.dirs(path = paste("best_runs", n, sep = "/"), recursive = FALSE) %>% 
  lapply(function(dir_name) paste(dir_name, "tfruns.d/flags.json", sep = "/")) %>% 
  lapply(yaml.load_file)

# load dataset directory paths
dataset_directory_paths <- list.dirs(paste("simulated_data", n, sep = "/"), recursive = FALSE)

for (dataset_directory_path in dataset_directory_paths) {
  dataset_path <- list.files(path = dataset_directory_path, pattern = "dataset.csv", full.names = TRUE)
  runs_dir_path <- paste(dataset_directory_path, "runs", sep = "/")
  
  counter <- 1
  for (flags in best_run_flags) {
    flags$dataset_path <- dataset_path
    training_run(file = "dataset_lstm_pred.R", flags = flags, run_dir = paste(runs_dir_path, counter, sep = "/"), echo = FALSE, view = FALSE)
    counter <- counter + 1
  }
  
  copy_run(ls_runs(order = "eval_mse_rowsum", decreasing = FALSE, runs_dir = runs_dir_path)[1,], to = paste(runs_dir_path, "best_run","single_step", sep = "/"))
  copy_run(ls_runs(order = "eval_mse_multistep_rowsum", decreasing = FALSE, runs_dir = runs_dir_path)[1,], to = paste(runs_dir_path, "best_run", "multi_step", sep = "/"))
}
