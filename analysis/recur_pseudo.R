
# R Script for data wrangling and analysis of cross validation

# importing
library(readr)
library(readxl)
# plotting
library(ggplot2) # https://www.ggplot2-exts.org/ ggplot extensions
# wrangling
library(tidyr)
library(dplyr)
library(tibble)
library(janitor)
library(stringr)
# programming
library(infer)
library(purrr)

library(plotROC)
library(ramify)
library(pROC)

# import labels
mosaic_labels <- read_csv("~/mosaic_list.csv")
patient_labels <- read_csv("~/patient_list.csv")

# factorize labels
factorize <- function(df){
  if (nrow(df) > 1000){
    df$ground <- as.factor(df$ground)
    df$pred <- as.factor(df$pred)
  } else {
    df$label <- as.factor(df$label)
    df$pred <- as.factor(df$pred)
  }
  return(df)
}

# import dataframe function
import_dataframe <- function(file){
  df <- read_excel(file, sheet = 1)
  return(df)
}

# function to be used to generate patient level labels
patient_label <- function(file){
  filename <- tail(str_split(file, "/")[[1]], n = 1)
  patient <- str_split(filename, "_")[[1]][1]
  return(patient)
}

# function to be used to generate mosaic level labels
mosaic_label <- function(file){
  filename <- tail(str_split(file, "/")[[1]], n = 1)
  patient <- str_split(filename, "_")[[1]][1]
  mosaic <- str_split(filename, "_")[[1]][2]
  return(str_c(patient, "_", mosaic))
}

# calculates accuracy
accuracy <- function(x_vec, y_vec){
    delta_fun <- function(x, y){ # kronecker delta function
      if (x == y){return(1)}
      else{return(0)}}
    # calculate the accuracy of vector, functional programming style
    val <- map2_dbl(x_vec, y_vec, delta_fun)
    return(sum(val)/length(x_vec))
}  

# import the entire kfold iteration ----
import_iteration <- function(filepath){
  df = data.frame()
  for (file in list.files(filepath)){
    print(file.path(filepath, file))
    df_iter <- import_dataframe(file.path(filepath, file))
    kthfold <- tail(str_split(filepath, "/")[[1]], n = 1)
    df_iter["kfold"] <- rep(kthfold, nrow(df_iter))
    df <- rbind(df, df_iter)
  }
  df["patient"] <- map_chr(df$filenames, patient_label)
  df["mosaic"] <- map_chr(df$filenames, mosaic_label)
  return(df)
}
kfold1_df <- import_iteration(filepath = "/Users/toddhollon/Desktop/Recur_pseudo/cv_round1")
kfold2_df <- import_iteration(filepath = "/Users/toddhollon/Desktop/Recur_pseudo/cv_round2")
kfold3_df <- import_iteration(filepath = "/Users/toddhollon/Desktop/Recur_pseudo/cv_round3")
kfold4_df <- import_iteration(filepath = "/Users/toddhollon/Desktop/Recur_pseudo/cv_round4")
kfold5_df <- import_iteration(filepath = "/Users/toddhollon/Desktop/Recur_pseudo/cv_round5")

# renormalize the softmaxes 
normalized_softmax_df <- function(df, level = 'mosaic'){
  # returns a dataframe with normalized softmax over either the mosaics or patients
  if (level == 'mosaic'){
    df <- group_by(df, mosaic)
  } else if (level == 'patient'){
    df <- group_by(df, patient)
  }
  
  # sum the values for each mosaic/patient
  unnormed_df <- df %>%
    summarise(nondiagnostic = sum(nondiagnostic),
              pseudoprogression = sum(pseudoprogression),
              recurrence = sum(recurrence))
  # renormalize the values of the pseudoprogressio and recurrence only 
  renormalize_softmax <- function(df){
    for (i in 1:nrow(df)){
      df[i,3:4] <- df[i,3:4]/sum(df[i,3:4]) # normalize across the softmax columns
    }
    return(df)
  }
  
  renorm_df <- renormalize_softmax(unnormed_df)
  renorm_df["kfold"] <- unique(df$kfold) # add column with the kfold label
  return(renorm_df)
}

# returns the final diagnosis based on thresholding
final_diagnosis <- function(df, label_df, recur_threshold = 0.50){
  diagnosis <- as.data.frame(ifelse(df$recurrence > recur_threshold, "recurrence", "pseudo"))
  new_df <- cbind(df, diagnosis)
  names(new_df)[6] <- "pred"
  new_df["label"] <- label_df$label
  
  new_df <- factorize(new_df)
  return(new_df)
}
mosaic_softmax_df <- final_diagnosis(mosaic_softmax_df, mosaic_labels)


# patch ROC curve----
# https://cran.r-project.org/web/packages/plotROC/vignettes/examples.html
ggplot(kfold1_df, aes(d = ground, m = recurrence)) + 
  geom_roc()
# mosaic ROC curve
ggplot(mosaic_softmax_df, aes(d = label, m = recurrence)) +
  geom_roc(n.cuts = 0)
# patient ROC curve
ggplot(patient_softmax_df, aes(d = label, m = recurrence)) +
  geom_roc()


# Working with the entire kfold dataset----
# combining all kfolds ----
kfold_list = list(kfold1_df, kfold2_df, kfold3_df, kfold4_df, kfold5_df) # list of kfolds
combine_kfolds <- function(df_list, label_df, threshold = 0.30, level = "mosaic"){
  for (i in seq_along(df_list)){
    if (level == "mosaic"){
      df_list[[i]] <- final_diagnosis(normalized_softmax_df(df_list[[i]], level = "mosaic"), 
                                      label_df = label_df,
                                      recur_threshold = threshold)
    } else if (level == "patient"){
      df_list[[i]] <- final_diagnosis(normalized_softmax_df(df_list[[i]], level = "patient"),
                                      label_df = label_df,
                                      recur_threshold = threshold)
    }
  }
  # concatenate each kfold_df into a single dataframe
  df = data.frame()
  for (i in seq_along(df_list)){
    df <- rbind(df_list[[i]], df)
  }
  return(df)
}
kfold_df_mosaic <- combine_kfolds(kfold_list, mosaic_labels)
kfold_df_patient <- combine_kfolds(kfold_list, patient_labels, level = "patient")

# Function based on the amount of probability on the mosaics
patient_prediction_via_mosaics <- function(df_mosaics, patient_df, recur_threshold = 0.62){
  df_predictions <- c() # initialize empty vector for predictions
  df_probabilities <- c()
  kfolds <- unique(df_mosaics$kfold) # find all the kfolds

  for (kfold_num in kfolds){
    df_kfold <- filter(df_mosaics, kfold == kfold_num) # build dataframe to collect each kfold
    patient_predictions <- c()
    recurrence_probs <- c()
    pseudo_probs <- c()
    for (patient in patient_df$patient){
      df <- df_kfold[str_detect(df_kfold$mosaic, patient),]
      recurrence_probs <- c(recurrence_probs, max(df$recurrence)) # maximum value of the probabilites
      pseudo_probs <- c(pseudo_probs, 1 - max(df$recurrence)) # maximum value of the probabilites
      if (any(df$recurrence > recur_threshold)){ # if any of the mosaics has a recurrence value > threshold
        patient_predictions <- c(patient_predictions, "recurrence")
      } else {
        patient_predictions <- c(patient_predictions, "pseudo")
      }
    }
  kfold <- rep(kfold_num, length(patient_predictions))
  df_predictions <- rbind(df_predictions, data.frame(patient_df$patient, recurrence_probs, pseudo_probs, patient_predictions, patient_df$label, kfold))
  }
  return(df_predictions)
}

# search probability threshold for optimal accuracy values
search_kfold_threshold <- function(){
  kfolds_acc <- c()
  indices <- c()
  for (kfold_num in unique(kfold_df_mosaic$kfold)){
    print(kfold_num)
    acc <- 0
    maxthresh <- 0
    for (thresh in seq(10, 90, 0.1)){
      df_kfold = filter(kfold_df_mosaic, kfold == kfold_num)
      df <- patient_prediction_via_mosaics(df_kfold, patient_labels, recur_threshold = thresh/100)
      cm <- confusionMatrix(as.factor(df$patient_predictions), as.factor(df$patient_df.label))
      thresh_acc <- cm$overall["Accuracy"]
      if (thresh_acc > acc){
        acc <- thresh_acc
        maxthresh <- thresh}
    }
    kfolds_acc <- c(kfolds_acc, acc)
    indices <- c(indices, maxthresh)
  }
  names(kfolds_acc) <- indices
  return(kfolds_acc)
}
kfold_threshold_results <- search_kfold_threshold()

acc <- c()
for (thresh in seq(10, 90, 0.1)){
    # df_kfold = filter(kfold_df_mosaic, kfold == kfold_num)
    df <- patient_prediction_via_mosaics(kfold_df_mosaic, patient_labels, recur_threshold = thresh/100)
    cm <- confusionMatrix(as.factor(df$patient_predictions), as.factor(df$patient_df.label))
    acc <- c(acc, cm$overall["Accuracy"])
}
names(acc) <- seq(10, 90, 0.1)
acc
patient_preds <- patient_prediction_via_mosaics(kfold_df_mosaic, patient_labels, recur_threshold = 0.62)

cm_kfold_iterator_mosaics <- function(df){
  acc = c()
  for (kfold_val in unique(kfold_df_mosaic$kfold)){
    kfold_filter <- filter(kfold_df_mosaic, kfold == kfold_val)
    pred_df <- patient_prediction_via_mosaics(kfold_filter, patient_labels, recur_threshold = 0.62)
    cm <- confusionMatrix(as.factor(pred_df$patient_predictions), as.factor(patient_labels$label))
    print(cm)
    acc <- c(acc, cm$overall["Accuracy"])
  }
  names(acc) <- unique(kfold_df_mosaic$kfold)
  print(acc)
} # will return the accuracy values as written
confusionMatrix(foo$patient_predictions, foo$patient_df.label)
cm_kfold_iterator_mosaics(patient_preds)

# table of predictions
kfold_pred_table <- function(df, patient_df, recur_threshold = 0.30){
  kfold_patients <- patient_prediction_via_mosaics(df, patient_df, recur_threshold = recur_threshold)# run function above 
  kfold_matrix <- as.data.frame.matrix(table(kfold_patients$patient_df.patient, kfold_patients$kfold)) # generate a matrix patient vs kfolds
  kfold_matrix["patients"] <- row.names(kfold_matrix)
  
  kfolds <- unique(kfold_patients$kfold)
  for (kfold_num in kfolds){
    df_kfold <- filter(kfold_patients, kfold == kfold_num) 
    kfold_matrix[kfold_num] <- df_kfold$patient_predictions == df_kfold$patient_df.label
  }
  return(kfold_matrix[,1:5])
} 
truth_matrix <- kfold_pred_table(kfold_df_mosaic, patient_labels)
ggplot(truth_matrix, aes())
# table of probabilities
kfold_prob_table_probs <- function(df, patient_df, recur_threshold = 0.62){
  kfold_patients <- patient_prediction_via_mosaics(df, patient_df, recur_threshold = recur_threshold)# run function above 
  str(kfold_patients)
  kfold_matrix <- as.data.frame.matrix(table(kfold_patients$patient_df.patient, kfold_patients$kfold)) # generate a matrix patient vs kfolds
  kfold_matrix["patients"] <- row.names(kfold_matrix)
  
  kfolds <- unique(kfold_patients$kfold)
  for (kfold_num in kfolds){
    df_kfold <- filter(kfold_patients, kfold == kfold_num) 
    kfold_matrix[kfold_num] <- df_kfold$recurrence
  }
  return(kfold_matrix[,1:5])
} 
probs_matrix = kfold_prob_table_probs(kfold_df_mosaic, patient_labels)

patient_prediction_via_mosaics_allkfolds <- function(kfold_matrix){
  truth_vect <- c()
  for (i in 1:nrow(kfold_matrix)){
    print(as.logical(kfold_matrix[i, 1:5]))
    truth_vect <- c(truth_vect, any(as.logical(kfold_matrix[i, 1:5])))
  }
  return(data.frame(kfold_matrix$patients, truth_vect))
}
foo <- patient_prediction_via_mosaics_allkfolds(patient_prediction_via_mosaics_allkfolds)

# Function based on the amount of probability on the patches
patient_prediction_via_patches <- function(df_patches, patient_df, ratio_threshold = 0.02, recur_threshold = 0.70){
  patient_predictions <- c()
  for (patient in patient_df$patient){
    df_patient <- df_patches[str_detect(df_patches$mosaic, patient),] # filter by patients
    mosaics <- unique(df_patient$mosaic)
    counter = 0
    for (mosaic in mosaics){
      df_mosaic <- df_patient[str_detect(df_patient$mosaic, mosaic),] # filter by mosaics
      if (counter == 0){
        recur <- sum(df_mosaic$recurrence > recur_threshold)
        if (recur/nrow(df_mosaic) > ratio_threshold){
          patient_predictions <- c(patient_predictions, "recurrence")
          counter <- counter + 1
        } else {
          patient_predictions <- c(patient_predictions, "pseudo")
          break
        }
      }  
    }
  }
  return(data.frame(patient_df$patient, patient_predictions, patient_df$label))
}
foo <- patient_prediction_via_patches(kfold1_df, patient_labels)
full_df <- rbind(kfold1_df, kfold2_df, kfold3_df, kfold4_df, kfold5_df)
cm_kfold_iterator_patches <- function(df){
  for (kfold_val in unique(df$kfold)){
    kfold_filter <- filter(df, kfold == kfold_val)
    pred_df <- patient_prediction_via_patches(kfold_filter, patient_labels)
    cm <- confusionMatrix(as.factor(pred_df$patient_predictions), as.factor(patient_labels$label))
    # print(cm$overall["Accuracy"])
    print(cm)
  }
} # will return the accuracy values as written 
cm_kfold_iterator_patches(full_df)
