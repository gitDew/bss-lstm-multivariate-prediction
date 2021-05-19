# Pairing function for unique seeds
pair <- function(x,y) {
  0.5*(x+y)*(x+y+1) +  x
}

save_to_directory <- function(dir_path, filename, dataset, S, A) {
  
  if (!dir.exists(dir_path)) {
    dir.create(dir_path)
  }
  
  write.csv(dataset, file = paste0(dir_path, filename, "_dataset.csv"))
  write.csv(S, file = paste0(dir_path, filename, "_source.csv"))
  write.csv(A, file = paste0(dir_path, filename, "_mixing_matrix.csv"))
}