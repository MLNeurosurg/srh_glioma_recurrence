
# R code for plotting results of prediction

# Project plotting---------------------
# check the incorrect cases
patch_df <- full_df[full_df$ground != 0,]
patch_df$ground = as.factor(patch_df$ground)

kfold_auc <- function(df, labels_name, predictions_name){
  rocs <- c()
  kfolds <- unique(df$kfold)
  for (kfold_num in kfolds){
    print(kfold_num)
    df_kfold <- filter(df, kfold == kfold_num) 
    kfold_roc = roc(df_kfold[[labels_name]],df_kfold[[predictions_name]])
    rocs <- c(rocs, auc(kfold_roc))
  }
  names(rocs) <- kfolds
  return(rocs)
}
# patch level AUC
kfold_auc(patch_df, labels_name = "ground", predictions_name = "recurrence")
# mosiac-level AUC
kfold_auc(kfold_df_mosaic, labels_name = "label", predictions_name = "recurrence")
# patient-level AUC
kfold_auc(patient_preds, labels_name = "patient_df.label", predictions_name = "recurrence_probs")

# coloring functionstr()
scale_fill_kfold <- function(...){
  ggplot2:::manual_scale('col', 
                         values = setNames(c("#999999", "#E69F00", "#56B4E9", "#7CAE00","#C77CFF"),
                                           c("cv_round2", "cv_round3", "cv_round4", "cv_round5", "cv_round6")),
                         ...
  )
}
# ROC for patches
ggplot(patch_df, aes(d = ground, m = recurrence, col = kfold)) + 
  geom_roc(cutoffs.at = c(0.5)) +
  style_roc()+
  scale_fill_kfold()  
# ROC for mosaics
ggplot(kfold_df_mosaic, aes(d = label, m = recurrence, col = kfold)) + 
  geom_roc(cutoffs.at = c(0.5)) +
  style_roc() +
  scale_fill_kfold()
# ROC for patients
ggplot(patient_preds, aes(d = patient_df.label, m = recurrence_probs, col = kfold)) + 
  geom_roc(cutoffs.at = c(0.5)) +
  style_roc() +
  scale_fill_kfold()

# patient plots prediction
patient_recur <- patient_preds[patient_preds$patient_df.label == "recurrence",]
patient_pseudo <- patient_preds[patient_preds$patient_df.label == "pseudo",]

# sort patients by label, then by patient
patient_preds <- patient_preds[with(patient_preds,
                                    order(patient_preds$patient_df.label,
                                          patient_preds$patient_df.patient)),]

rownames(patient_preds) <- c()
patient_preds["study_num"] <- rep(seq_len(35), each=5)

rownames(patient_preds) <- c()
ggplot(patient_preds, aes(x = study_num, y = recurrence_probs, col = kfold)) + 
  geom_point(size = 1) + 
  ylim(c(0, 1)) + 
  scale_x_continuous(breaks=seq(1, 35, 1)) + 
  coord_flip() + 
  scale_fill_kfold() +
  stat_summary(fun.y = mean, geom = "point", col = "black", alpha = 0.8, size = 0.75) +
  stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1),
               geom = "errorbar", width = 0.05, col = "black", alpha = 0.8) +
  theme(axis.text.x = element_text(angle = 90, hjust = 0.5))

ggplot(patient_pseudo, aes(x = patient_df.patient, y = recurrence_probs, col = kfold)) + 
  geom_point(size = 1) + 
  ylim(c(0, 1)) + 
  stat_summary(fun.y = mean, geom = "point", col = "black", alpha = 0.8, size = 0.75) +
  stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1),
               geom = "errorbar", width = 0.1, col = "black", alpha = 0.8) +
  theme(axis.text.x = element_text(angle = 90, hjust = 0.5))

kfold_df <- kfold_df_mosaic[kfold_df_mosaic$label == "recurrence",]
ggplot(kfold_df, aes(x = mosaic, y = recurrence, col = kfold)) + 
  geom_point() + 
  # facet_grid(.~label) + 
  stat_summary(fun.y = mean, geom = "point", col = "black", alpha = 0.8) +
  stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1),
               geom = "errorbar", width = 0.3, col = "black", alpha = 0.8) +
  theme(axis.text.x = element_text(angle = 90, hjust = 0.5))



# TSNE visualization--------
tsne_df <- read_excel("/Users/toddhollon/Desktop/tsne_df.xlsx")
ggplot(tsne_df, aes(x = xs, y = ys, col = labels)) +
  geom_point(size = 0.5) + 
  theme_bw() +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")) 
  
