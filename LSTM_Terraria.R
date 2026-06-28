library(dplyr)
library(tidyr)
library(lubridate)
library(ggplot2)
library(tidyverse)
library(keras3)
library(tensorflow)
library(zoo)
#install_tensorflow()

# ── 0. Random Seed ──────────────────────────────────────────────
set.seed(42)
tensorflow::tf$random$set_seed(42)

# ── 1. Read the CSV file ───────────────────────────────────────
df <- read_csv("Normalized Data/norm_Terraria_ccu.csv") %>%
  mutate(DateTime = as.Date(DateTime)) %>%
  arrange(DateTime)

cat("The number of data:", nrow(df), "\n")
cat("Period:", as.character(min(df$DateTime)), "〜", as.character(max(df$DateTime)), "\n")
cat("The number of Events:", sum(df$Label), "\n\n")

# ── 2. Parameters ───────────────────────────────────
LOOKBACK    <- 30      # Input
HORIZON     <- 7       # Prediction (7 means 1 week)
N_FEATURES  <- 4       # Players, Final, Label, Major Update
BATCH_SIZE  <- 64
EPOCHS      <- 200
LSTM_UNITS  <- 64
DROPOUT     <- 0.2
VALID_RATIO <- 0.1     # The ratio for Validation
TEST_DAYS   <- 90      # Testing data (90days)

# ── 3. Sequence ───────────────────────────────────────

feature_mat <- df %>%
  select(Players, Final, Label,Major_Update) %>%
  as.matrix()

target_vec <- df$Players

make_sequences <- function(features, targets, lookback, horizon) {
  n <- nrow(features)
  max_i <- n - lookback - horizon + 1
  
  X <- array(NA, dim = c(max_i, lookback, ncol(features)))
  Y <- matrix(NA, nrow = max_i, ncol = horizon)
  
  for (i in seq_len(max_i)) {
    X[i, , ] <- features[i:(i + lookback - 1), ]
    Y[i, ]   <- targets[(i + lookback):(i + lookback + horizon - 1)]
  }
  list(X = X, Y = Y)
}

seqs <- make_sequences(feature_mat, target_vec, LOOKBACK, HORIZON)
X_all <- seqs$X
Y_all <- seqs$Y

cat("The Number of Sample:", nrow(X_all), "\n")
cat("X shape:", paste(dim(X_all), collapse = " x "), "\n")
cat("Y shape:", paste(dim(Y_all), collapse = " x "), "\n\n")

# ── 4. Train / Validation / Test  ──────────────────────
n_total <- nrow(X_all)
n_test  <- TEST_DAYS
n_val   <- round((n_total - n_test) * VALID_RATIO)
n_train <- n_total - n_test - n_val

idx_train <- 1:n_train
idx_val   <- (n_train + 1):(n_train + n_val)
idx_test  <- (n_train + n_val + 1):n_total

X_train <- X_all[idx_train, , , drop = FALSE]
Y_train <- Y_all[idx_train, , drop = FALSE]
X_val   <- X_all[idx_val,   , , drop = FALSE]
Y_val   <- Y_all[idx_val,   , drop = FALSE]
X_test  <- X_all[idx_test,  , , drop = FALSE]
Y_test  <- Y_all[idx_test,  , drop = FALSE]

cat(sprintf("Train: %d  Val: %d  Test: %d\n\n", n_train, n_val, n_test))

# ── 5. Construct the Model ───────────────────────────────────────────
build_model <- function(lookback, n_features, horizon,
                        lstm_units = 64, dropout = 0.2) {
  input <- layer_input(shape = c(lookback, n_features))
  
  x <- input %>%
    layer_lstm(units = lstm_units, return_sequences = TRUE,   #Additional Layer
               kernel_regularizer = regularizer_l2(1e-4)) %>% #Additional Layer
    layer_dropout(rate = dropout) %>%                         #Additional Layer
    layer_lstm(units = lstm_units %/% 2, return_sequences = TRUE,
               kernel_regularizer = regularizer_l2(1e-4)) %>%
    layer_dropout(rate = dropout) %>%
    layer_lstm(units = lstm_units %/% 2, return_sequences = FALSE,
               kernel_regularizer = regularizer_l2(1e-4)) %>%
    layer_dropout(rate = dropout) %>%
    layer_dense(units = 32, activation = "tanh") %>%
    layer_dense(units = horizon)   # Generate
  
  model <- keras_model(inputs = input, outputs = x)
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 1e-3),
    loss      = "mse",
    metrics   = list("mae")
  )
  model
}

model <- build_model(LOOKBACK, N_FEATURES, HORIZON, LSTM_UNITS, DROPOUT)
summary(model)

# ── 6. Learning ────────────────────────────────────────────────
callbacks_list <- list(
  callback_early_stopping(
    monitor  = "val_loss",
    patience = 15,
    restore_best_weights = TRUE
  ),
  callback_reduce_lr_on_plateau(
    monitor  = "val_loss",
    factor   = 0.5,
    patience = 7,
    min_lr   = 1e-6
  ),
  callback_model_checkpoint(
    filepath         = "lstm/best_lstm_Terraria_5layers_4features.keras",
    monitor          = "val_loss",
    save_best_only   = TRUE
  )
)

history <- model %>% fit(
  x          = X_train,
  y          = Y_train,
  batch_size = BATCH_SIZE,
  epochs     = EPOCHS,
  validation_data = list(X_val, Y_val),
  callbacks  = callbacks_list,
  verbose    = 1
)

# ── 7. Evaluation(MSE & MAE) ────────────────────────────────────────────────
eval_result <- model %>% evaluate(X_test, Y_test, verbose = 0)
cat(sprintf("\nTest Loss (MSE): %.6f\nTest MAE:        %.6f\n\n",
            eval_result["loss"], eval_result["mae"]))
cat(sprintf("\n%.6f\n%.6f\n\n",
            eval_result["loss"], eval_result["mae"]))

# ── 8. Prediction & Visualization ──────────────────────────
Y_pred <- model %>% predict(X_test, verbose = 0)

# The period for Testing
date_offset     <- n_train + n_val + LOOKBACK
test_start_row  <- date_offset + 1             
test_dates      <- df$DateTime[test_start_row:(test_start_row + n_test - 1)]


pred_step1 <- Y_pred[, 1]
true_step1 <- Y_test[, 1]

plot_df <- tibble(
  Date      = test_dates,
  Actual    = true_step1,
  Predicted = pred_step1
)

p <- ggplot(plot_df, aes(x = Date)) +
  geom_line(aes(y = Actual,    color = "Actual value"),    linewidth = 0.8) +
  geom_line(aes(y = Predicted, color = "LSTM predicted value"), linewidth = 0.8,
            linetype = "dashed") +
  scale_color_manual(values = c("Actual value" = "#2196F3",
                                "LSTM predicted value" = "#F44336")) +
  labs(
    title    = "Terraria Steam CCU prediction (LSTM) — Testing Period",
    subtitle = sprintf("Next-day prediction | Lookback=%d days | LSTM units=%d",
                       LOOKBACK, LSTM_UNITS),
    x        = "Date",
    y        = "Normalized concurrent users",
    color    = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "top")

print(p)
ggsave("plot/Terraria5layers_lstm_test_prediction.png", p, width = 10, height = 5, dpi = 150)
cat("Saved: Terraria5layers_lstm_test_prediction.png\n")

# ── 9. Learning curve ─────────────────────────────────────
hist_df <- tibble(
  epoch    = seq_along(history$metrics$loss),
  train    = history$metrics$loss,
  val      = history$metrics$val_loss
)

p2 <- ggplot(hist_df, aes(x = epoch)) +
  geom_line(aes(y = train, color = "Train Loss")) +
  geom_line(aes(y = val,   color = "Val Loss")) +
  scale_color_manual(values = c("Train Loss" = "#4CAF50",
                                "Val Loss"   = "#FF9800")) +
  labs(title = "Learning curve (MSE Loss)", x = "Epoch", y = "Loss", color = NULL) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "top")

print(p2)
ggsave("plot/Terraria_5layers_lstm_training_curve.png", p2, width = 8, height = 4, dpi = 150)
cat("Saved: Terraria_5layers_lstm_training_curve.png\n")

# ── 10. The Furure Prediction ────────────
last_window <- feature_mat[(nrow(feature_mat) - LOOKBACK + 1):nrow(feature_mat), ]
last_window_arr <- array(last_window, dim = c(1, LOOKBACK, N_FEATURES))

future_pred <- model %>% predict(last_window_arr, verbose = 0)
future_dates <- max(df$DateTime) + seq_len(HORIZON)

future_df <- tibble(
  Date      = future_dates,
  Predicted = as.vector(future_pred)
)

cat("\n── In the future", HORIZON, "Daily Prediction ──\n")
print(future_df)

# ── 11. Save the model ─────────────────────────────────────────
model %>% save_model("lstm/lstm_Terraria_5layers_final.keras")
cat("\n Saved: lstm_Terraria_5layersfinal.keras\n")

# ── Example ────────────────────────────
# loaded_model <- load_model("lstm/lstmTerraria_5layers_final.keras")
