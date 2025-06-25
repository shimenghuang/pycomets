# Experiment for Dudu IV
# LK 2025

set.seed(42)

### DEPs
library("comets")

### DGP PARs
b <- c(0.5, 1.0, 0.5, 1.0, 0.5)
a1 <- c(1.5, 1.0)
a2 <- c(1.0, 1.5)
vc1 <- toeplitz(2:1 / 2)
vc2 <- toeplitz(3:1 / 3)

### DGP
dgp <- function(n, beta = b, alpha1 = a1, alpha2 = a2, theta = 1) {
    H <- matrix(rnorm(2 * n, mean = 1), ncol = 2, nrow = n)
    V <- H + mvtnorm::rmvnorm(n, sigma = vc1)
    W <- mvtnorm::rmvnorm(n, sigma = vc2)
    D <- c(cbind(V, W) %*% beta) + c(H %*% alpha1) + rnorm(n)
    Y <- D * theta + c(H %*% alpha2) + rnorm(n)
    list(H = H, V = V, W = W, D = D, Y = Y)
}

### SIM PARs
nsim <- 300
n <- 300
reg <- "lrm"
tt <- "quadratic"

### RUN
pb <- txtProgressBar(0, nsim, style = 3, width = 60)
res <- lapply(seq_len(nsim), \(iter) {
    setTxtProgressBar(pb, iter)
    dd <- dgp(n)
    gcm <- comets(Y ~ V + W | H + D,
        data = dd, reg_YonZ = reg, reg_XonZ = reg, type = tt, B = 499
    )
    pcm <- comets(Y ~ V + W | H + D,
        test = "pcm", data = dd, reg_YonXZ = reg, reg_YonZ = reg,
        reg_YhatonZ = reg, reg_VonXZ = reg, reg_RonZ = reg,
        est_vhat = FALSE
    )
    data.frame(
        gcm = gcm$p.value,
        pcm = pcm$p.value
    )
}) |> do.call("rbind", args = _)

### VIS
plot(ecdf(res[, "gcm"]), col = "darkblue", main = "", xlab = "p-value")
plot(ecdf(res[, "pcm"]), add = TRUE, col = "darkred")
legend("topleft", c("GCM", "PCM"), col = c("darkblue", "darkred"), lwd = 2, bty = "n")
abline(0, 1)

hist(res[, "gcm"])

### Load saved samples
library(dplyr)
nsim <- 100
reg <- "lrm"
tt <- "quadratic"
pb <- txtProgressBar(0, nsim, style = 3, width = 60)
res <- lapply(seq_len(nsim), \(iter) {
    setTxtProgressBar(pb, iter)
    mat <- read.csv(paste0("pycomets/inst/tmp_simulated_data/sim", iter - 1, ".csv"))
    df <- as.data.frame(mat)
    dd <- list(
        H = df %>% dplyr::select(starts_with("H")) %>% as.matrix(),
        V = df %>% dplyr::select(starts_with("V")) %>% as.matrix(),
        W = df %>% dplyr::select(starts_with("W")) %>% as.matrix(),
        D = df$D,
        Y = df$Y
    )
    gcm <- comets(Y ~ V + W | H + D,
        data = dd, reg_YonZ = reg, reg_XonZ = reg, type = tt, B = 499
    )
    pcm <- comets(Y ~ V + W | H + D,
        test = "pcm", data = dd, reg_YonXZ = reg, reg_YonZ = reg,
        reg_YhatonZ = reg, reg_VonXZ = reg, reg_RonZ = reg,
        est_vhat = FALSE
    )
    data.frame(
        gcm = gcm$p.value,
        pcm = pcm$p.value
    )
}) |> do.call("rbind", args = _)

### VIS
plot(ecdf(res[, "gcm"]), col = "darkblue", main = "", xlab = "p-value")
plot(ecdf(res[, "pcm"]), add = TRUE, col = "darkred")
legend("topleft", c("GCM", "PCM"), col = c("darkblue", "darkred"), lwd = 2, bty = "n")
abline(0, 1)

hist(res[, "gcm"])
