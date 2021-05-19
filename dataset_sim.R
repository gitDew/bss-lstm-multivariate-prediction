source('SOBI_pred_fxns.R')
source('utils.R')

ns <- as.numeric(commandArgs(trailingOnly = TRUE)) # ns <- c(200, 500, 2000, 5000)

foldername <- paste(ns, collapse = "_")
output_dir <- paste0("simulated_data/", ns)

logfile <- paste0(output_dir, "/log.txt")

if (!dir.exists(output_dir)) {
  dir.create(output_dir)  
}

repetitions <- 2000
max_lookback <- 12

h <- c(1, 6)
k <- c(1, 2, 12)
bss_names <- c("SOBI", "AMUSE")
forecast_names <- c("ARIMA", "ETS")

JADE_params <- expand.grid(h=h, k=k, bss_method=bss_names, forecast_method=forecast_names, stringsAsFactors = FALSE)

# number of rows for measurements table space pre-allocation
num_rows <- repetitions * length(ns) * (nrow(JADE_params) + length(h)) # + length(h) for VAR predictions
n_col <- numeric(num_rows)
i_col <- numeric(num_rows)
h_col <- numeric(num_rows)
k_col <- numeric(num_rows)
bss_col <- character(num_rows)
forecast_col <- character(num_rows)
mse_col <- numeric(num_rows)
c <- 1

tryCatch(
  expr = {
    for (n in ns) {
      for (i in 1:repetitions) {
        
        dir_path <- paste(output_dir, paste(n, i, sep="_"),"", sep="/")
        
        if (dir.exists(dir_path)) {
          next
        }
        
        # set unique seed for each iteration
        set.seed(pair(n, i))
        
        A <- matrix(rnorm(25),5,5)
        
        # source signals with unit variances
        # AR(2)
        s1 <- arima.sim(n=n, list(ar=c(0.3639, 0.5941)), sd=0.3563)
        # AR(3)
        s2 <- arima.sim(n=n, list(ar=c(0.1159,0.3853,0.1756)), sd=0.8449)
        # MA(8)
        s3 <- arima.sim(n=n, list(ma=c(0.1216, 0.9385, 0.9409, 0.9144, 0.3823, 0.6862, 0.8894, 0.7561)), sd=0.4227)
        # AR(5)
        s4 <- arima.sim(n=n, list(ar=c(0.199, 0.2152, 0.153, 0.1435, 0.151)), sd=0.7118)
        # AR(7)
        s5 <- arima.sim(n=n, list(ar=c(0.1452, 0.05336, 0.08183, 0.1739, 0.06431, 0.2536, 0.1341)), sd=0.6991)
        
        S <- cbind(s1, s2, s3, s4, s5)
      
        dataset <- tcrossprod(S, A)
      
        # add artificial means
        dataset <- sweep(dataset, 2, c(1,2,3,4,5), "+")
        
        # normalize data
        test_set_size <- max_lookback + max(h) # for fair comparison with NNs
        train_data <- head(dataset, -test_set_size)
        mean <- apply(train_data, 2, mean)
        std <- apply(train_data, 2, sd)
        dataset <- scale(dataset, center = mean, scale = std)
        
        
        # JADE predictions
        for (row_index in 1:nrow(JADE_params)) {
          row <- as.list(JADE_params[row_index,])
          tryCatch(
            expr = {
              split_index <- n - row$h
              
              training_set <- dataset[1:split_index,]
              testing_set <- dataset[-(1:split_index),]
              
              arguments <- c(list(data=training_set), row)
              
              i_col[c] <- i
              n_col[c] <- n
              h_col[c] <- arguments$h
              k_col[c] <- arguments$k
              bss_col[c] <- arguments$bss_method
              forecast_col[c] <- arguments$forecast_method

              pred <- do.call(predict_with_JADE, arguments)
              
              mse <- mean(rowSums((testing_set-pred)**2))
              mse_col[c] <- mse
            },
            error = function(e) {
              write(paste(i, n, paste(row, collapse = ", "), toString(e)), logfile, append = TRUE)
            },
            finally = {
              c <- c + 1
            }
          )
        }
        
        # VAR predictions
        for (step in h) {
          tryCatch(
            expr = {
              split_index <- n - step
              
              training_set <- dataset[1:split_index,]
              testing_set <- dataset[-(1:split_index),]
              
              # predict keeps warning about se.fit not working for the multivariate case, even though it is set to false
              suppressWarnings(VAR_preds <- predict(ar(training_set, se.fit = FALSE),n.ahead=step))
              
              VAR_mse <- mean(rowSums((testing_set - VAR_preds$pred)**2))
              
              i_col[c] <- i
              n_col[c] <- n
              h_col[c] <- step
              k_col[c] <- NA
              bss_col[c] <- NA
              forecast_col[c] <- "VAR"
              mse_col[c] <- VAR_mse
            },
            error = function(e) {
              write(paste(paste(i, n, "VAR"), toString(e)), logfile, append = TRUE)
            },
            finally = {
              c <- c + 1
            }
          )
          

        }
        save_to_directory(dir_path, paste(n, i, sep="_"), dataset, S, A)
      }
    }
    
  },
  
  interrupt = function(e) {
    message("Process interrupted, stopping...")
  },
  
  finally = {
    measurements <- data.frame(n=n_col, i=i_col, h=h_col, k=k_col, bss=bss_col, forecast=forecast_col, mse=mse_col, stringsAsFactors = FALSE)
    saveRDS(measurements, file = paste0(output_dir, "/measurements.rds"))
    message("Done, quitting.")
  }
)
