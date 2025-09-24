# loading trajectory data and carry out LLM analysis

library(rhdf5)
result_folder <- "/Users/binghao/Desktop/Research/OT-singlecell/UOTReg/results/"
file = "dynamics/embryoid/traj_bundle.h5"
file_path = paste(result_folder, file, sep = "")

h5ls(file_path)  # optional: see contents

traj_reg   <- h5read(file_path, "traj_reg")    # array [9, N, 1000]
gene_names <- h5read(file_path, "gene_names")  # character vector length 1000
lab_reg_km <- h5read(file_path, "lab_reg_km")  # character or numeric vector length 1000

# Quick checks
dim(traj_reg)   # should be c(9, N, 1000)
length(gene_names); length(lab_reg_km)
head(gene_names); head(lab_reg_km)

###########################
# Visually checking genes #
library(dplyr)
library(ggplot2)
library(tidyr)

# time points (original, for plotting)
time_vec_orig <- seq(1.5, 25.5, by = 3)  # 1.5, 4.5, ..., 25.5
N <- dim(traj_reg)[2]
Tm <- dim(traj_reg)[3]
stopifnot(Tm == length(time_vec_orig), length(lab_reg_km) == N)

cell_ids <- paste0("cell", seq_len(N))
cluster_f <- factor(as.integer(lab_reg_km))
cl_levels <- levels(cluster_f)

# ---- function to plot one gene quickly ----
plot_gene_quick <- function(gene_pattern = "SOX2", n_per_cluster = 10,
                            facet_scales = "fixed", seed = 1) {
  # fuzzy match (allow suffixes/prefixes)
  hits <- which(grepl(gene_pattern, gene_names, ignore.case = FALSE))
  if (length(hits) == 0) stop("No gene matched '", gene_pattern, "'.")
  g_idx <- hits[1]
  g_name <- gene_names[g_idx]
  
  # build long df for ALL cells (for the mean curve)
  mat_all <- traj_reg[g_idx, , , drop = FALSE][1, , ]  # [cells x time]
  df_all <- data.frame(
    cell    = rep(cell_ids, each = Tm),
    cluster = rep(cluster_f, each = Tm),
    time    = rep(time_vec_orig, times = N),
    expr    = as.numeric(t(mat_all))
  )
  df_all$cluster <- droplevels(df_all$cluster)
  
  # sample up to n_per_cluster cells per cluster for the spaghetti lines
  set.seed(seed)
  samp_ids <- df_all %>%
    dplyr::distinct(cell, cluster) %>%
    dplyr::group_by(cluster) %>%
    dplyr::mutate(.rand = runif(dplyr::n())) %>%
    dplyr::slice_min(.rand, n = n_per_cluster, with_ties = FALSE) %>%
    dplyr::ungroup() %>%
    dplyr::pull(cell)
  
  df_samp <- df_all %>% filter(cell %in% samp_ids)
  
  # mean (over all cells in cluster) at each time
  df_mean <- df_all %>%
    group_by(cluster, time) %>%
    summarise(mu = mean(expr), .groups = "drop")
  
  message(sprintf("Plotting '%s' | sampled %d cells total (%d per cluster max).",
                  g_name, length(unique(df_samp$cell)), n_per_cluster))
  
  # plot: spaghetti of sampled cells (gray), plus cluster mean (colored)
  ggplot() +
    geom_line(data = df_samp,
              aes(x = time, y = expr, group = cell),
              linewidth = 0.4, alpha = 0.35, color = "gray40") +
    geom_point(data = df_samp,
               aes(x = time, y = expr, group = cell),
               size = 0.6, alpha = 0.35, color = "gray40") +
    geom_line(data = df_mean,
              aes(x = time, y = mu, color = cluster, group = cluster),
              linewidth = 1.1) +
    geom_point(data = df_mean,
               aes(x = time, y = mu, color = cluster),
               size = 1.6) +
    scale_color_discrete(drop = FALSE) +
    facet_wrap(~ cluster, scales = facet_scales) +  # set to "free_y" if ranges differ a lot
    labs(
      title = paste0(g_name, " — expression over time by cluster"),
      subtitle = "Thin gray lines: 10 sampled trajectories/cluster; colored line: cluster mean (all cells)",
      x = "Time", y = "Expression", color = "Cluster"
    ) +
    theme_classic() +
    theme(strip.background = element_blank(),
          panel.grid = element_blank())
}

# ---- Example: SOX2 ----
plot_gene_quick("SMAGP", n_per_cluster = 10, facet_scales = "fixed")
# If clusters differ wildly in dynamic range, try:

## ------------------------------------------------------------
## Allow for flexible model for each cluster
## ------------------------------------------------------------

## ============================================================
## Per-cluster model choice (1p / 2p / 3p) for a single gene
##   - Fits *within each cluster* using only its cells
##   - One-piece:      expr ~ t_cent + (1 + t_cent | cell)
##   - Two-piece:      expr ~ (t_cent + hinge[tau]) + (1 + t_cent | cell)
##   - Three-piece:    expr ~ (t_cent + h1[tau1] + h2[tau2]) + (1 + t_cent | cell)
##   - Visualizes mean±SE (solid) + FE fit (dashed) for all clusters on one plot
##   - Draws cluster-colored vertical lines at the estimated break(s)
## ============================================================

library(lme4)
library(lmerTest)
library(dplyr)
library(tidyr)
library(ggplot2)

# ---- Basics & checks ----
stopifnot(length(dim(traj_reg)) == 3)
Gp <- dim(traj_reg)[1]; N <- dim(traj_reg)[2]; Tm <- dim(traj_reg)[3]
stopifnot(Tm == 9, length(lab_reg_km) == N)

time_vec_orig <- seq(1.5, 25.5, by = 3)            # 1.5, 4.5, ..., 25.5
t_centered    <- time_vec_orig - mean(time_vec_orig)

cell_ids <- paste0("cell", seq_len(N))

# ---- Choose gene (fuzzy match) ----
gene_pat <- "PAX6"  # <-- change here
hit <- which(grepl(gene_pat, gene_names, fixed = TRUE))
stopifnot(length(hit) >= 1)
g_idx <- hit[1]

# ---- Build long DF for that gene across ALL clusters ----
# traj_reg[gene, cell, time] -> [N x 9]
mat_all <- traj_reg[g_idx, , , drop = FALSE][1, , ]
cluster_f <- factor(as.integer(lab_reg_km))  # 1..K clusters

df_all <- data.frame(
  time_orig = rep(time_vec_orig, times = N),
  t_cent    = rep(t_centered,    times = N),
  cell      = rep(cell_ids,      each  = length(time_vec_orig)),
  cluster   = rep(cluster_f,     each  = length(time_vec_orig)),
  expr      = as.numeric(t(mat_all))
)
df_all$cell    <- factor(df_all$cell)
df_all$cluster <- droplevels(df_all$cluster)
cl_levels <- levels(df_all$cluster)

# ---- Tell the script which model each cluster should use
# Options: "1p", "2p", "3p".
# Provide a named vector; any missing names default to "1p".
model_map <- setNames(rep("1p", length(cl_levels)), cl_levels)
# Example custom choices (edit to your needs):
model_map["5"] <- "1p"   # cluster 4 -> three-piece
model_map["1"] <- "1p"   # cluster 2 -> two-piece

# ---- Helpers ----
fit_lmer_safe <- function(formula, data) {
  m <- try(lmer(formula, data = data, REML = TRUE), silent = TRUE)
  if (inherits(m, "try-error") || isSingular(m)) {
    f_alt <- update(formula, . ~ . - (1 + t_cent | cell) + (1 | cell))
    m <- lmer(f_alt, data = data, REML = TRUE)
  }
  m
}

# Grid candidates (interior times) for change points (centered scale)
cand <- t_centered[2:(length(t_centered)-1)]       # 7 interior points

fit_2p_best <- function(df_cl) {
  # try all taus; choose lowest AIC
  try_one <- function(tau) {
    df_cl$hinge <- pmax(0, df_cl$t_cent - tau)
    m <- fit_lmer_safe(expr ~ t_cent + hinge + (1 + t_cent | cell), df_cl)
    list(AIC = AIC(m), tau = tau, m = m)
  }
  res <- lapply(cand, try_one)
  res[[ which.min(sapply(res, `[[`, "AIC")) ]]
}

fit_3p_best <- function(df_cl) {
  # try all tau1<tau2; choose lowest AIC
  pairs_tau <- t(combn(cand, 2))
  try_pair <- function(tau1, tau2) {
    df_cl$h1 <- pmax(0, df_cl$t_cent - tau1)
    df_cl$h2 <- pmax(0, df_cl$t_cent - tau2)
    m <- fit_lmer_safe(expr ~ t_cent + h1 + h2 + (1 + t_cent | cell), df_cl)
    list(AIC = AIC(m), tau1 = tau1, tau2 = tau2, m = m)
  }
  res <- apply(pairs_tau, 1, \(pr) try_pair(pr[1], pr[2]))
  res[[ which.min(sapply(res, `[[`, "AIC")) ]]
}

to_orig_time <- function(tau_c) time_vec_orig[ which.min(abs(t_centered - tau_c)) ]

# ---- Fit per cluster according to model_map ----
fits_per_cluster <- list()
pred_lines_list  <- list()
break_lines_df   <- list()  # for per-cluster vertical lines

for (cl in cl_levels) {
  df_cl <- df_all %>% filter(cluster == cl)
  this_model <- ifelse(is.na(model_map[cl]), "1p", model_map[cl])
  
  if (this_model == "1p") {
    m <- fit_lmer_safe(expr ~ t_cent + (1 + t_cent | cell), df_cl)
    b <- fixef(m)
    # prediction on grid
    fe_line <- data.frame(
      time_orig = time_vec_orig,
      t_cent    = t_centered,
      cluster   = cl,
      pred      = as.numeric(b["(Intercept)"] + b["t_cent"] * t_centered)
    )
    fits_per_cluster[[cl]] <- list(model = "1p", m = m, slopes = c(s1 = unname(b["t_cent"])))
    pred_lines_list[[cl]]  <- fe_line
    break_lines_df[[cl]]   <- NULL
    
  } else if (this_model == "2p") {
    best2 <- fit_2p_best(df_cl)
    m <- best2$m; tau <- best2$tau
    b <- fixef(m)
    # slopes pre/post
    s_pre  <- unname(b["t_cent"])
    s_post <- unname(b["t_cent"] + b["hinge"])
    # prediction on grid
    hinge <- pmax(0, t_centered - tau)
    fe_line <- data.frame(
      time_orig = time_vec_orig,
      t_cent    = t_centered,
      cluster   = cl,
      pred      = as.numeric(b["(Intercept)"] + b["t_cent"] * t_centered + b["hinge"] * hinge)
    )
    fits_per_cluster[[cl]] <- list(model = "2p", m = m, tau = tau,
                                   slopes = c(s_pre = s_pre, s_post = s_post))
    pred_lines_list[[cl]]  <- fe_line
    break_lines_df[[cl]]   <- data.frame(cluster = cl, tau_time = to_orig_time(tau))
    
  } else if (this_model == "3p") {
    best3 <- fit_3p_best(df_cl)
    m <- best3$m; tau1 <- best3$tau1; tau2 <- best3$tau2
    b <- fixef(m)
    # segment slopes
    s1 <- unname(b["t_cent"])
    s2 <- unname(b["t_cent"] + b["h1"])
    s3 <- unname(b["t_cent"] + b["h1"] + b["h2"])
    # prediction on grid
    h1 <- pmax(0, t_centered - tau1)
    h2 <- pmax(0, t_centered - tau2)
    fe_line <- data.frame(
      time_orig = time_vec_orig,
      t_cent    = t_centered,
      cluster   = cl,
      pred      = as.numeric(b["(Intercept)"] + b["t_cent"]*t_centered + b["h1"]*h1 + b["h2"]*h2)
    )
    fits_per_cluster[[cl]] <- list(model = "3p", m = m, tau1 = tau1, tau2 = tau2,
                                   slopes = c(s1 = s1, s2 = s2, s3 = s3))
    pred_lines_list[[cl]]  <- fe_line
    break_lines_df[[cl]]   <- rbind(
      data.frame(cluster = cl, tau_time = to_orig_time(tau1)),
      data.frame(cluster = cl, tau_time = to_orig_time(tau2))
    )
    
  } else {
    stop(sprintf("Unknown model '%s' for cluster %s. Use '1p','2p','3p'.", this_model, cl))
  }
}

# ---- Summaries for plotting ----
df_summ <- df_all %>%
  group_by(cluster, time_orig) %>%
  summarise(mu = mean(expr), se = sd(expr)/sqrt(n()), .groups = "drop")

fe_line_all <- bind_rows(pred_lines_list)
break_lines_all <- bind_rows(Filter(Negate(is.null), break_lines_df))

# ---- Plot: observed mean±SE per cluster + cluster-specific fitted curve ----
p <- ggplot() +
  geom_line(data = df_summ, aes(time_orig, mu, color = cluster, group = cluster), linewidth = 0.9) +
  geom_point(data = df_summ, aes(time_orig, mu, color = cluster), size = 1.7) +
  geom_errorbar(data = df_summ, aes(time_orig, ymin = mu - se, ymax = mu + se, color = cluster),
                width = 0.35, alpha = 0.7) +
  geom_line(data = fe_line_all, aes(time_orig, pred, color = cluster, group = cluster),
            linewidth = 1.1, linetype = 2) +
  labs(
    title    = paste0(gene_names[g_idx], " — per-cluster chosen models"),
    subtitle = "Solid: observed mean±SE | Dashed: cluster-specific fixed-effects fit",
    x = "Time", y = "Expression", color = "Cluster"
  ) +
  theme_classic()

# Add cluster-colored break lines if any 2p/3p models were used
if (nrow(break_lines_all) > 0) {
  p <- p + geom_vline(data = break_lines_all,
                      aes(xintercept = tau_time, color = cluster),
                      linetype = 3, alpha = 0.5)
}

print(p)

##### Saving data for visualization #####

# --------- R SAVE SIDE ----------

# # Save per-gene payload for Python WITHOUT changing cluster labels
# save_lmm_viz_json <- function(
#     gene_name,
#     df_summ,          # columns: cluster, time_orig, mu, se
#     fe_line_all,      # columns: cluster, time_orig, pred
#     break_lines_all,  # columns: cluster, tau_time  (can be empty)
#     model_map,        # named vec/list: names are clusters, values "1p"/"2p"/"3p"
#     outfile
# ){
#   stopifnot(requireNamespace("dplyr", quietly = TRUE),
#             requireNamespace("jsonlite", quietly = TRUE))
#   
#   as_int <- function(x) as.integer(as.character(x))
#   
#   df_summ2 <- df_summ |>
#     dplyr::mutate(cluster = as_int(cluster)) |>
#     dplyr::arrange(cluster, time_orig)
#   
#   fe_line2 <- fe_line_all |>
#     dplyr::mutate(cluster = as_int(cluster)) |>
#     dplyr::arrange(cluster, time_orig)
#   
#   if (!is.null(break_lines_all) && nrow(break_lines_all) > 0) {
#     breaks2 <- break_lines_all |>
#       dplyr::mutate(cluster = as_int(cluster)) |>
#       dplyr::distinct(cluster, tau_time) |>
#       dplyr::arrange(cluster, tau_time)
#   } else {
#     breaks2 <- data.frame()
#   }
#   
#   mm_tbl <- tibble::tibble(
#     cluster = as.integer(names(model_map)),
#     model   = as.character(unname(model_map))
#   ) |>
#     dplyr::arrange(cluster)
#   
#   payload <- list(
#     gene           = gene_name,
#     cluster_levels = sort(unique(df_summ2$cluster)),
#     df_summ        = df_summ2,
#     fe_line        = fe_line2,
#     breaks         = breaks2,
#     model_map      = mm_tbl
#   )
#   
#   result_folder = "/Users/binghao/Desktop/Research/OT-singlecell/UOTReg/results/dynamics/embryoid/LMM"
#   jsonlite::write_json(
#     payload,
#     path        = file.path(result_folder, outfile),
#     dataframe   = "rows",
#     pretty      = TRUE,
#     auto_unbox  = TRUE,
#     na          = "null"
#   )
# }
# 
# # ---- example call (COL1A1) ----
# genenow = "CLDN4"
# save_lmm_viz_json(
#   gene_name  = genenow,
#   df_summ    = df_summ,
#   fe_line_all = fe_line_all,
#   break_lines_all = break_lines_all,
#   model_map  = model_map,           # if you have it; otherwise pass setNames(rep("1p", 6), 0:5)
#   outfile    = paste(genenow,"_lmm_viz.json",sep="")
# )

