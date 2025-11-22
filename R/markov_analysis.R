# Analisis Rantai Markov untuk Perpindahan Merek Laptop Mahasiswa
# Versi R - Mirip dengan Notebook Python
# Tanggal: 22 November 2025

# Setup R Environment
# install.packages(c("tidyverse", "markovchain", "igraph", "expm", "ggplot2", "reshape2", "gridExtra"), repos = "https://cran.rstudio.com/")
library(tidyverse)
library(markovchain)
library(igraph)
library(expm)
library(ggplot2)
library(reshape2)
library(gridExtra)

# Atur tema plot
theme_set(theme_minimal() + theme(plot.title = element_text(hjust = 0.5, face = "bold")))

# 1. Load dan Bersihkan Data
data <- read.csv("../data/datapemstok.csv", encoding = "latin1")

# Select relevant columns
data_clean <- data %>%
  select(
    first_brand = "Apa.merek.laptop.yang.pertama.kali.Anda.gunakan.saat.menjadi.mahasiswa",
    still_same = "Apakah.saat.ini.Anda.masih.menggunakan.laptop.yang.sama.",
    current_brand = "Apabila.Anda.sudah.pernah.berganti.laptop..apa.merek.laptop.Anda.saat.ini."
  )

# Handle logic
data_clean <- data_clean %>%
  mutate(current_brand = ifelse(still_same == "Ya", first_brand, current_brand)) %>%
  drop_na(first_brand, current_brand)

print("Pratinjau data bersih:")
print(head(data_clean))

# 2. Identifikasi Ruang Keadaan
states <- sort(unique(c(data_clean$first_brand, data_clean$current_brand)))
state_index <- setNames(seq_along(states), states)

print("Ruang Keadaan:")
print(states)
print(paste("Jumlah keadaan:", length(states)))

# 3. Matriks Transisi
trans_table <- table(data_clean$first_brand, data_clean$current_brand)
transition_matrix <- prop.table(trans_table, 1)

# Build Markov Chain Model
markov_model <- new("markovchain", states = states, byrow = TRUE,
                    transitionMatrix = transition_matrix,
                    name = "Laptop Brand Switching")

print("Matriks Transisi:")
print(transition_matrix)

# Visualisasi heatmap
trans_df <- melt(transition_matrix, varnames = c("From", "To"), value.name = "Probability")

ggplot(trans_df, aes(x = To, y = From, fill = Probability)) +
  geom_tile(color = "white", size = 0.5) +
  scale_fill_gradient(low = "white", high = "steelblue", na.value = "white") +
  labs(title = "Heatmap Matriks Transisi Rantai Markov",
       x = "Keadaan Tujuan (Merek Saat Ini)",
       y = "Keadaan Awal (Merek Awal)",
       fill = "Probabilitas Transisi") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(size = 16, face = "bold"))

ggsave("../output/transition_matrix_heatmap_academic_R.png", width = 12, height = 10, dpi = 600)

# 4. Diagram Transisi
g <- graph_from_adjacency_matrix(transition_matrix, mode = "directed", weighted = TRUE)

# Filter edges with low probability
E(g)$weight_filtered <- ifelse(E(g)$weight > 0.01, E(g)$weight, 0)
g_filtered <- delete_edges(g, E(g)[E(g)$weight_filtered == 0])

layout <- layout_in_circle(g_filtered)

# Color nodes: absorbing in red, others in blue
absorbing_states <- states[diag(transition_matrix) == 1 & rowSums(transition_matrix) == 1]
node_colors <- ifelse(V(g_filtered)$name %in% absorbing_states, "coral", "lightblue")

plot(g_filtered, layout = layout,
     vertex.color = node_colors,
     vertex.size = 3000,
     vertex.label.cex = 1.2,
     vertex.label.color = "black",
     edge.arrow.size = 0.8,
     edge.width = E(g_filtered)$weight * 15,
     edge.color = "gray",
     main = "Diagram Transisi Rantai Markov\nPerpindahan Merek Laptop Mahasiswa")

# Add legend
legend("topright", legend = c("Keadaan Penyerap", "Keadaan Lain"),
       fill = c("coral", "lightblue"), bty = "n")

png("../output/transition_diagram_academic_R.png", width = 1200, height = 1000, res = 150)
plot(g_filtered, layout = layout,
     vertex.color = node_colors,
     vertex.size = 3000,
     vertex.label.cex = 1.2,
     vertex.label.color = "black",
     edge.arrow.size = 0.8,
     edge.width = E(g_filtered)$weight * 15,
     edge.color = "gray",
     main = "Diagram Transisi Rantai Markov\nPerpindahan Merek Laptop Mahasiswa")
legend("topright", legend = c("Keadaan Penyerap", "Keadaan Lain"),
       fill = c("coral", "lightblue"), bty = "n")
dev.off()

# Properti graf
print("Properti Graf:")
print(paste("Jumlah simpul:", vcount(g)))
print(paste("Jumlah busur:", ecount(g)))
print(paste("Derajat rata-rata:", mean(degree(g))))
print(paste("Terhubung kuat:", is_connected(g, mode = "strong")))
print(paste("Jumlah komponen terhubung kuat:", count_components(g, mode = "strong")))

# 5. Probabilitas Langkah ke-n
n_steps <- c(3, 5, 10)

for (n in n_steps) {
  P_n <- transition_matrix %^% n
  print(paste("Probabilitas setelah", n, "langkah:"))
  print(P_n)
  
  # Visualization for Lenovo
  if ("Lenovo" %in% states) {
    lenovo_idx <- which(states == "Lenovo")
    probs <- P_n[lenovo_idx, ]
    
    df_plot <- data.frame(State = states, Probability = probs)
    
    p <- ggplot(df_plot, aes(x = State, y = Probability)) +
      geom_bar(stat = "identity", fill = "cornflowerblue", alpha = 0.8, color = "black", width = 0.6) +
      geom_text(aes(label = sprintf("%.3f", Probability)), vjust = -0.5, size = 3.5) +
      labs(title = paste("Probabilitas Transisi Langkah ke-n\nDistribusi setelah", n, "Langkah Dimulai dari Lenovo"),
           x = "Keadaan Tujuan", y = "Probabilitas") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1),
            plot.title = element_text(size = 14, face = "bold")) +
      ylim(0, max(probs) * 1.15)
    
    print(p)
    ggsave(paste0("../output/n_step_probabilities_", n, "_steps_academic_R.png"), width = 10, height = 8, dpi = 600)
  }
}

# 6. Distribusi Stasioner
steady_state <- steadyStates(markov_model)

print("Distribusi Stasioner:")
for (i in seq_along(states)) {
  print(paste(states[i], ":", steady_state[i]))
}

# Visualization
df_steady <- data.frame(State = states, Probability = as.numeric(steady_state))
ggplot(df_steady, aes(x = State, y = Probability)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.8, color = "black", width = 0.6) +
  geom_text(aes(label = sprintf("%.3f", Probability)), vjust = -0.5, size = 3.5) +
  labs(title = "Distribusi Stasioner Rantai Markov\nPrediksi Pangsa Pasar Jangka Panjang",
       x = "Merek Laptop", y = "Probabilitas Stasioner") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(size = 16, face = "bold")) +
  ylim(0, max(steady_state) * 1.15)

ggsave("../output/stationary_distribution_academic_R.png", width = 12, height = 10, dpi = 600)

# Verifikasi
check <- steady_state %*% transition_matrix
print("Verifikasi πP ≈ π:")
print(all.equal(as.numeric(steady_state), as.numeric(check)))
print(paste("Perbedaan maksimum:", max(abs(steady_state - check))))

# 7. Klasifikasi Ruang Keadaan
absorbing_states <- states[diag(transition_matrix) == 1 & rowSums(transition_matrix) == 1]
recurrent_states <- setdiff(states, absorbing_states)  # Simplified
transient_states <- setdiff(states, c(absorbing_states, recurrent_states))

print("Klasifikasi Ruang Keadaan:")
print(paste("Keadaan Penyerap:", paste(absorbing_states, collapse = ", ")))
print(paste("Keadaan Berulang:", paste(recurrent_states, collapse = ", ")))
print(paste("Keadaan Sementara:", paste(transient_states, collapse = ", ")))

# Visualization klasifikasi
classification <- data.frame(
  State = states,
  Type = case_when(
    State %in% absorbing_states ~ "Absorbing",
    State %in% recurrent_states ~ "Recurrent",
    TRUE ~ "Transient"
  )
)

colors_class <- c("Absorbing" = "#d62728", "Recurrent" = "#2ca02c", "Transient" = "#ff7f0e")

ggplot(df_steady, aes(x = State, y = Probability, fill = classification$Type)) +
  geom_bar(stat = "identity", alpha = 0.8, color = "black", width = 0.6) +
  scale_fill_manual(values = colors_class) +
  labs(title = "Distribusi Stasioner dengan Klasifikasi Keadaan\nAnalisis Loyalitas Merek Laptop",
       x = "Merek Laptop", y = "Probabilitas Stasioner", fill = "Tipe Keadaan") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(size = 16, face = "bold"),
        legend.position = "top")

ggsave("../output/state_classification_academic_R.png", width = 12, height = 10, dpi = 600)

# Analisis tambahan
print("Analisis Tambahan:")
print(paste("Jumlah komponen terhubung:", count_components(g)))
print(paste("Graf terhubung:", is_connected(g)))
print(paste("Graf strongly connected:", is_connected(g, mode = "strong")))

# Simulasi
set.seed(42)
simulasi <- rmarkovchain(n = 15, object = markov_model, t0 = "Lenovo")
print("Simulasi jalur dari Lenovo selama 15 langkah:")
print(simulasi)

# Ekspor hasil
write.csv(transition_matrix, "../output/transition_matrix_R.csv")
write.csv(df_steady, "../output/stationary_distribution_R.csv", row.names = FALSE)
write.csv(classification, "../output/state_classification_R.csv", row.names = FALSE)

print("Hasil utama diekspor ke file CSV di direktori output.")
# Transition Diagram
g <- graph_from_adjacency_matrix(transition_matrix, mode = "directed", weighted = TRUE)

# Filter edges with low probability
E(g)$weight_filtered <- ifelse(E(g)$weight > 0.01, E(g)$weight, 0)
g_filtered <- delete_edges(g, E(g)[E(g)$weight_filtered == 0])

layout <- layout_in_circle(g_filtered)

# Color nodes: absorbing in red, others in blue
absorbing_states <- states[diag(transition_matrix) == 1 & rowSums(transition_matrix) == 1]
node_colors <- ifelse(V(g_filtered)$name %in% absorbing_states, "coral", "lightblue")

plot(g_filtered, layout = layout,
     vertex.color = node_colors,
     vertex.size = 3000,
     vertex.label.cex = 1.2,
     vertex.label.color = "black",
     edge.arrow.size = 0.8,
     edge.width = E(g_filtered)$weight * 15,
     edge.color = "gray",
     main = "Diagram Transisi Rantai Markov\nPerpindahan Merek Laptop Mahasiswa")

# Add legend
legend("topright", legend = c("Keadaan Penyerap", "Keadaan Lain"),
       fill = c("coral", "lightblue"), bty = "n")

png("../output/transition_diagram_academic_R.png", width = 1200, height = 1000, res = 150)
plot(g_filtered, layout = layout,
     vertex.color = node_colors,
     vertex.size = 3000,
     vertex.label.cex = 1.2,
     vertex.label.color = "black",
     edge.arrow.size = 0.8,
     edge.width = E(g_filtered)$weight * 15,
     edge.color = "gray",
     main = "Diagram Transisi Rantai Markov\nPerpindahan Merek Laptop Mahasiswa")
legend("topright", legend = c("Keadaan Penyerap", "Keadaan Lain"),
       fill = c("coral", "lightblue"), bty = "n")
dev.off()

# Probability at n-th step
n_steps <- c(3, 5, 10)

for (n in n_steps) {
  P_n <- transition_matrix %^% n
  print(paste("Probabilitas setelah", n, "langkah:"))
  print(P_n)
  
  # Visualization for Lenovo
  if ("Lenovo" %in% states) {
    lenovo_idx <- which(states == "Lenovo")
    probs <- P_n[lenovo_idx, ]
    
    df_plot <- data.frame(State = states, Probability = probs)
    
    p <- ggplot(df_plot, aes(x = State, y = Probability)) +
      geom_bar(stat = "identity", fill = "cornflowerblue", alpha = 0.8, color = "black", width = 0.6) +
      geom_text(aes(label = sprintf("%.3f", Probability)), vjust = -0.5, size = 3.5) +
      labs(title = paste("Probabilitas Transisi Langkah ke-n\nDistribusi setelah", n, "Langkah Dimulai dari Lenovo"),
           x = "Keadaan Tujuan", y = "Probabilitas") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1),
            plot.title = element_text(size = 14, face = "bold")) +
      ylim(0, max(probs) * 1.15)
    
    print(p)
    ggsave(paste0("../output/n_step_probabilities_", n, "_steps_academic_R.png"), width = 10, height = 8, dpi = 600)
  }
}

# Steady-State Distribution
steady_state <- steadyStates(markov_model)
print("Distribusi Stasioner:")
for (i in seq_along(states)) {
  print(paste(states[i], ":", steady_state[i]))
}

# Visualization
df_steady <- data.frame(State = states, Probability = as.numeric(steady_state))
ggplot(df_steady, aes(x = State, y = Probability)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.8, color = "black", width = 0.6) +
  geom_text(aes(label = sprintf("%.3f", Probability)), vjust = -0.5, size = 3.5) +
  labs(title = "Distribusi Stasioner Rantai Markov\nPrediksi Pangsa Pasar Jangka Panjang",
       x = "Merek Laptop", y = "Probabilitas Stasioner") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(size = 16, face = "bold")) +
  ylim(0, max(steady_state) * 1.15)

ggsave("../output/stationary_distribution_academic_R.png", width = 12, height = 10, dpi = 600)

# Verifikasi
check <- steady_state %*% transition_matrix
print("Verifikasi πP ≈ π:")
print(all.equal(as.numeric(steady_state), as.numeric(check)))
print(paste("Perbedaan maksimum:", max(abs(steady_state - check))))

# State Classification
absorbing_states <- states[diag(transition_matrix) == 1 & rowSums(transition_matrix) == 1]
recurrent_states <- setdiff(states, absorbing_states)  # Simplified
transient_states <- setdiff(states, c(absorbing_states, recurrent_states))

print("Klasifikasi Ruang Keadaan:")
print(paste("Keadaan Penyerap:", paste(absorbing_states, collapse = ", ")))
print(paste("Keadaan Berulang:", paste(recurrent_states, collapse = ", ")))
print(paste("Keadaan Sementara:", paste(transient_states, collapse = ", ")))

# Visualization klasifikasi
classification <- data.frame(
  State = states,
  Type = case_when(
    State %in% absorbing_states ~ "Absorbing",
    State %in% recurrent_states ~ "Recurrent",
    TRUE ~ "Transient"
  )
)

colors_class <- c("Absorbing" = "#d62728", "Recurrent" = "#2ca02c", "Transient" = "#ff7f0e")

ggplot(df_steady, aes(x = State, y = Probability, fill = classification$Type)) +
  geom_bar(stat = "identity", alpha = 0.8, color = "black", width = 0.6) +
  scale_fill_manual(values = colors_class) +
  labs(title = "Distribusi Stasioner dengan Klasifikasi Keadaan\nAnalisis Loyalitas Merek Laptop",
       x = "Merek Laptop", y = "Probabilitas Stasioner", fill = "Tipe Keadaan") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(size = 16, face = "bold"),
        legend.position = "top")

ggsave("../output/state_classification_academic_R.png", width = 12, height = 10, dpi = 600)

# Analisis tambahan
print("Analisis Tambahan:")
print(paste("Jumlah komponen terhubung:", count_components(g)))
print(paste("Graf terhubung:", is_connected(g)))
print(paste("Graf strongly connected:", is_connected(g, mode = "strong")))

# Simulation
set.seed(42)
simulasi <- rmarkovchain(n = 15, object = markov_model, t0 = "Lenovo")
print("Simulasi jalur dari Lenovo selama 15 langkah:")
print(simulasi)

# Ekspor hasil
write.csv(transition_matrix, "../output/transition_matrix_R.csv")
write.csv(df_steady, "../output/stationary_distribution_R.csv", row.names = FALSE)
write.csv(classification, "../output/state_classification_R.csv", row.names = FALSE)

print("Hasil utama diekspor ke file CSV di direktori output.")