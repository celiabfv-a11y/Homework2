---
title: "Statistical Learning - Homework 2: Supervised Learning"
author: "Malen Abarrategui Meire and Celia Benavente Fernández de Velasco"
date: "December 2025"
output:
  html_document:
    theme: cerulean
    highlight: tango
    number_sections: no
    toc: no
  pdf_document:
    theme: cerulean
    highlight: tango
    number_sections: yes
    toc: yes
editor_options:
  chunk_output_type: console
---

```{r global_options, include=T, echo = F}
knitr::opts_chunk$set(echo = T, warning=FALSE, message=FALSE)
```

# Introduction

The objective is to predict the presence of heart disease (target) using a variety of clinical and demographic features from the heart.csv dataset. This analysis focuses on data preprocessing, model training (Random Forest for prediction, Logistic Regression for interpretation), and feature selection.

```{r}
# Set the working directory 
# This line is user-specific and may need adjustment
# setwd()

# Load essential libraries for data science workflow    # Data manipulation and visualization
library(caret)        # Model training, cross-validation, and evaluation
library(pROC)         # ROC curves and AUC calculation
library(glmnet)       # Penalized regression (Lasso/Ridge)
library(rpart)
library(rpart.plot)# Decision Trees
library(randomForest)
library(MASS)
library(tidyverse)# Random Forest

# Load the heart disease dataset
heart_data = read.csv("heart.csv", header = TRUE, sep = ",")
```

# Data Preprocessing and Visualization

## Feature Engineering and Conversion

The target variable is target (1 = heart disease, 0 = no heart disease). Several other variables are categorical/ordinal and need to be converted to the factor type for proper modeling.

```{r}
# List of all categorical columns including the target
categorical_cols = c("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", 
                     "thal", "target")

# Convert listed columns to factors using the mutate and across functions from tidyverse
heart_data_clean = heart_data %>%
  mutate(across(all_of(categorical_cols), as.factor))

# Rename target levels for clarity
heart_data_clean$target = factor(heart_data_clean$target, 
                                 levels = c("0", "1"),
                                 labels = c("NoDisease", "Disease"))

# Display structure and check for missing data
glimpse(heart_data_clean)
summary(heart_data_clean)
# Check for missing values (NA)
sum(is.na(heart_data_clean))
```

## Exploratory Data Analysis

Visualization to understand the relationship between key predictors and the target variable.

```{r}
# Target distribution by Age using density plots
heart_data_clean %>%
  ggplot(aes(x = age, fill = target)) +
  geom_density(alpha = 0.6) +
  labs(title = "Heart Disease Distribution by Age",
       x = "Age (years)",
       fill = "Heart Disease") +
  theme_minimal()
```

Observation: The density plot indicates that the distribution of patients with heart disease ('Disease') is slightly concentrated towards younger ages compared to those without ('NoDisease').

```{r}
# Target distribution by Maximum Heart Rate Achieved (thalach)
heart_data_clean %>%
  ggplot(aes(x = thalach, fill = target)) +
  geom_density(alpha = 0.6) +
  labs(title = "Heart Disease Distribution by Max Heart Rate",
       x = "Maximum Heart Rate (thalach)",
       fill = "Heart Disease") +
  theme_minimal()
```

Observation: Patients with heart disease generally achieved a higher maximum heart rate (thalach), suggesting this is a strong predictor differentiating the two groups.

```{r}
# Target distribution by Chest Pain Type (cp)
heart_data_clean %>%
  ggplot(aes(x = cp, fill = target)) +
  geom_bar(position = "fill") + # position="fill" shows proportions
  labs(title = "Heart Disease Proportion by Chest Pain Type (cp)",
       x = "Chest Pain Type",
       y = "Proportion",
       fill = "Heart Disease") +
  scale_fill_manual(values = c("NoDisease" = "blue", "Disease" = "red")) +
  theme_minimal()
```

Observation: Chest pain types 1, 2, and 3 show a much higher proportion of 'Disease' compared to type 4 (asymptomatic), confirming cp is a very strong predictor.

## Data Spliting and Preprocessing

Split the data into training (70%) and testing (30%) sets, and apply scaling only to numerical predictors.

```{r}
# Set seed for reproducibility
set.seed(123) 
# Create data partition index
inTrain = createDataPartition(y = heart_data_clean$target, p = .70, list = FALSE)
training = heart_data_clean[inTrain, ]
testing  = heart_data_clean[-inTrain, ]

# This prevents prediction errors when a factor level exists in the training set 
# but is missing in the test set (or vice versa).
categorical_predictors = categorical_cols[categorical_cols != "target"]

# Loop through all categorical predictors (excluding the target)
# and force the test set factors to use the same levels as the training set.
for(col in categorical_predictors){
  testing[[col]] = factor(testing[[col]], levels = levels(training[[col]]))
}

# Preprocessing: Center and Scale numerical predictors in the training set
preProcValues = preProcess(training %>% dplyr::select(-all_of(categorical_cols)), 
                           method = c("center", "scale"))

# Apply preprocessing (scaling numerics, keeping factors) to both sets
training_preproc = predict(preProcValues, training)
testing_preproc = predict(preProcValues, testing)
```

# Classification Models

We will use two models: Random Forest (for prediction emphasis) and Logistic Regression (for interpretation emphasis).

## Random Forest-Machine learning

Random Forest (RF) is used for high predictive accuracy. We tune its main hyperparameter, mtry.

```{r}
# Setup cross-validation control (5-fold CV)
ctrl_rf = trainControl(method = "cv", 
                       number = 5, 
                       classProbs = TRUE, 
                       summaryFunction = twoClassSummary)

# Train the Random Forest model
set.seed(123)
rf_model = train(target ~ ., 
                 data = training_preproc, 
                 method = "rf", 
                 metric = "ROC", # Optimize for Area Under the ROC Curve
                 trControl = ctrl_rf,
                 tuneLength = 5) # Test 5 different values for mtry

print(rf_model)
```

Random Forest Performance Evaluation

```{r}
# Predict probabilities on the test set for ROC/AUC
rf_pred_probs = predict(rf_model, testing_preproc, type = "prob")[, "Disease"]
# Predict class labels using the default threshold (0.5)
rf_pred_class = predict(rf_model, testing_preproc)

# Confusion Matrix and overall metrics
rf_cm = confusionMatrix(rf_pred_class, testing_preproc$target, positive = "Disease")
print(rf_cm)

# ROC Curve and AUC
rf_roc = roc(testing_preproc$target, rf_pred_probs)
auc_rf = auc(rf_roc)
cat(paste("Test Set AUC (Random Forest):", round(auc_rf, 4), "\n"))

# Plot the ROC curve
plot(rf_roc, main = "ROC Curve for Random Forest", col = "darkblue", lwd = 2)

```

## LDA -Probabilistic learning

Linear Discriminant Analysis (LDA) is a classical Bayesian-based classification method that seeks to find a linear combination of features that best separates two or more classes.Its goal in our project is predictive as well.

```{r}
lda.model <- lda(target ~ ., data = training_preproc)

# View the model(prior probabilities,coefficients)
print(lda.model)
#The most important variables in our coefficients seem to be with this model cp and ca.
# Predictions on the test set
lda_test_results <- predict(lda.model, newdata = testing_preproc)

# 1.Posterior probabilities
lda_probabilities <- lda_test_results$posterior
head(lda_probabilities)

# 2.Class prediction
lda_prediction_class <- lda_test_results$class
head(lda_prediction_class)
# Confusion matrix
lda_cm <- confusionMatrix(lda_prediction_class, testing_preproc$target, positive = "Disease")
print(lda_cm)

# ROC curve and AUC
# We use the column disease of the posterior probabilities
lda_roc <- roc(testing_preproc$target, lda_probabilities[, "Disease"])
auc_lda <- auc(lda_roc)

cat(paste("Test Set AUC (LDA):", round(auc_lda, 4), "\n"))
```

## Decision trees
A Decision Tree is a non-parametric supervised learning algorithm that partitions the data into smaller, homogeneous subsets through a series of hierarchical decision rules.

```{r}
# Hyper-parameters
control_dt_rpart = rpart.control(
  minsplit = 20,    # Minimum of observations
  maxdepth = 4,     # Maximum deepness
  cp = 0.005        # Complexity parameter
)

dtFit_rpart <- rpart(target ~ ., 
                     data = training_preproc, 
                     method = "class", 
                     control = control_dt_rpart)

summary(dtFit_rpart)
#The most important variables are cp,thalach and ca.
```

```{r}
#Decision tree visualization
rpart.plot(dtFit_rpart, type = 4, extra = 101, fallen.leaves = TRUE, main = "Factors of heart disease")
#Evaluation of decision tree

# Prediction of probabilities in the test set
dt_pred_probs_rpart <- predict(dtFit_rpart, testing_preproc, type = "prob")[, "Disease"]

# Prediction of classes
dt_pred_class_rpart <- factor(ifelse(dt_pred_probs_rpart > 0.5, "Disease", "NoDisease"),
                              levels = levels(training_preproc$target))

# Confusion matrix
dt_cm_rpart <- confusionMatrix(dt_pred_class_rpart, testing_preproc$target, positive = "Disease")
print(dt_cm_rpart)

# ROC curve
dt_roc_rpart <- roc(testing_preproc$target, dt_pred_probs_rpart)
auc_dt_rpart <- auc(dt_roc_rpart)
cat(paste("Test Set AUC (Decision Tree - rpart):", round(auc_dt_rpart, 4), "\n"))
```

## Logistic Regression

Logistic Regression provides clear interpretability through its coefficients (log-odds).

```{r}
# Train Logistic Regression using the full set of preprocessed variables
logit_model = glm(target ~ ., 
                  data = training_preproc, 
                  family = "binomial") # Specify binomial family for binary classification

summary(logit_model)
```

Logistic Regression Interpretation

```{r}
# Calculate odds ratios (exponentiate coefficients)
odds_ratios = exp(coef(logit_model))
print("Odds Ratios:")
print(round(odds_ratios, 3))
```

Interpretation:

-   Odds Ratio \> 1.0: An increase in the predictor value or switching to that factor level (compared to the reference level) is associated with an increase in the odds of having heart disease. For example, Chest Pain Type 2 and 3 show very high odds ratios, confirming their strong positive link to the disease.

-   Odds Ratio \< 1.0: Associated with a decrease in the odds of having heart disease.

-   Counter-intuitive Example (exang): The odds ratio for exang (exercise-induced angina) is often less than 1, suggesting it reduces the odds of heart disease (target=1). This is clinically unexpected and often warrants further investigation, as it may be due to confounding effects from other correlated variables (e.g., patients with exang=1 might be managed more aggressively or have less advanced disease overall in this specific dataset).

Logistic Regression Performance

```{r}
# Predict probabilities on the test set
logit_pred_probs = predict(logit_model, testing_preproc, type = "response")
# Convert probabilities to class labels using the default 0.5 threshold
logit_pred_class = factor(ifelse(logit_pred_probs > 0.5, "Disease", "NoDisease"),
                          levels = c("NoDisease", "Disease"))

# Confusion Matrix and overall metrics
logit_cm = confusionMatrix(logit_pred_class, testing_preproc$target, positive = "Disease")
print(logit_cm)

# ROC Curve and AUC
logit_roc = roc(testing_preproc$target, logit_pred_probs)
auc_logit = auc(logit_roc)
cat(paste("Test Set AUC (Logistic Regression):", round(auc_logit, 4), "\n"))
```

## Threshold Optimization

We optimize the threshold for the Logistic Regression model based on a hypothetical cost matrix, an approach derived from risk learning.

Assume a hypothetical Cost/Benefit Matrix (Profit/Loss per patient):

```{r}
# Define the profit/cost matrix:
# TN: 0 (No action, correct) | FN: -5 (Missed case, high cost)
# FP: -1 (Unnecessary test/treatment, low cost) | TP: 2 (Correct diagnosis/successful retention, gain)
profit_unit = matrix(c(0, -5, # TN, FN
                       -1, 2), # FP, TP
                     nrow = 2, byrow = TRUE, 
                     dimnames = list(c("Predict NoDisease", "Predict Disease"),
                                     c("Reference NoDisease", "Reference Disease")))

thresholds = seq(0.05, 0.95, 0.05)
profit_i = matrix(NA, nrow = 1, ncol = length(thresholds), 
                  dimnames = list("Profit", thresholds))

# Calculate expected profit for each threshold on the test set
for (j in 1:length(thresholds)) {
  threshold = thresholds[j]
  
  # Predict based on the new threshold
  pred_class = factor(ifelse(logit_pred_probs > threshold, "Disease", "NoDisease"),
                      levels = c("NoDisease", "Disease"))
  
  # Confusion Matrix
  CM = confusionMatrix(pred_class, testing_preproc$target, positive = "Disease")$table
  
  # Calculate average profit per patient: total profit / total patients
  profit_applicant = sum(profit_unit * CM) / sum(CM)
  profit_i[1, j] = profit_applicant
}

# Visualize profit vs. threshold
barplot(profit_i, 
        main = "Average Profit vs. Classification Threshold",
        ylab = "Average Profit per Patient",
        xlab = "Threshold",
        col = "skyblue")

# Find the threshold that maximizes profit
optimal_threshold = thresholds[which.max(profit_i)]
max_profit = max(profit_i)

cat(paste("Optimal Threshold:", optimal_threshold, "\n"))
cat(paste("Maximum Average Profit per Patient:", round(max_profit, 4), "\n"))
```

Conclusion: The optimal threshold is often lower than the default 0.5 when the cost of a False Negative (missed case) is high, as shown by this result, which prioritizes minimizing the highly penalized FN error.

# Feature Selection and Comparison of Predictor Sets

## Feature Selection

We use the Variable Importance from the Random Forest model (a non-linear model) to select a reduced set of key predictors for a simpler Logistic Regression model.

```{r feature_selection_corrected}
# Get variable importance from Random Forest
rf_var_imp = varImp(rf_model, scale = FALSE)$importance

# Sort and select the top 10 most important features (selecting more helps ensure factor columns are captured)
top_features_raw = rownames(rf_var_imp)[order(rf_var_imp$Overall, decreasing = TRUE)][1:10]
cat("Top 10 RAW Feature Names (may contain dummy variables):\n", top_features_raw, "\n")

# --- CORRECTION: CLEAN FEATURE NAMES FOR GLM ---
# The goal is to get the base variable name (e.g., 'cp' from 'cp4') for glm
factor_vars = c("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal")

# Function to strip dummy variable suffixes, keeping only the base factor name
clean_feature_name = function(feature_name) {
  for (base_name in factor_vars) {
    # Check if the feature name starts with a factor name and has a suffix
    if (startsWith(feature_name, base_name) && nchar(feature_name) > nchar(base_name)) {
      return(base_name)
    }
  }
  return(feature_name) # Returns numerical or already clean categorical variables
}

# Apply the cleaning function to the raw list
top_features_all = sapply(top_features_raw, clean_feature_name)

# Select the top 5 unique features (to avoid duplicating factor names)
top_features = unique(top_features_all)[1:5]

cat("Top 5 FINAL Feature Names for GLM:\n", top_features, "\n")
```

## Model Training with Reduced Predictor Set

Train a new Logistic Regression model using only the top 5 selected features.

```{r}
# Create the formula for the reduced model (e.g., target ~ thal + cp + ca + thalach + oldpeak)
reduced_formula = as.formula(paste("target ~", paste(top_features, collapse = " + ")))
print(reduced_formula)

# Train the reduced Logistic Regression model
reduced_logit_model = glm(reduced_formula, 
                          data = training_preproc, 
                          family = "binomial")
summary(reduced_logit_model)
```

## Comparison of Predictor Sets

Compare the performance (AUC, Accuracy) of the full Logistic Regression model and the reduced one.

```{r}
# Predictions from reduced model
reduced_logit_pred_probs = predict(reduced_logit_model, testing_preproc, type = "response")
reduced_logit_pred_class = factor(ifelse(reduced_logit_pred_probs > 0.5, "Disease", "NoDisease"),
                                  levels = c("NoDisease", "Disease"))
# AUC for reduced model
reduced_logit_roc = roc(testing_preproc$target, reduced_logit_pred_probs)
auc_reduced_logit = auc(reduced_logit_roc)

# Compare Results in a single data frame
results_comparison = data.frame(
  Model = c("Random Forest", "LDA Model", "Decision Tree (rpart)", "Full Logit Model", "Reduced Logit Model"),
  AUC = c(auc_rf, auc_lda, auc_dt_rpart, auc_logit, auc_reduced_logit),
  Accuracy = c(rf_cm$overall['Accuracy'], 
               lda_cm$overall['Accuracy'], 
               dt_cm_rpart$overall['Accuracy'], 
               logit_cm$overall['Accuracy'],
               confusionMatrix(reduced_logit_pred_class, testing_preproc$target, positive = "Disease")$overall['Accuracy'])
)


results_comparison = results_comparison[order(-results_comparison$AUC), ]
print(results_comparison)

# Visualize ROC curves for comparison

plot(rf_roc, col = "darkblue", lwd = 2, main = "Final Model Comparison: ROC Curves")
plot(logit_roc, col = "red", lwd = 2, add = TRUE)
plot(reduced_logit_roc, col = "green4", lwd = 2, add = TRUE)
plot(lda_roc, col = "purple", lwd = 2, add = TRUE)
plot(dt_roc_rpart, col = "orange", lwd = 2, add = TRUE)


legend("bottomright", 
       legend = c(paste("Random Forest (AUC:", round(auc_rf, 3), ")"), 
                  paste("Full Logit (AUC:", round(auc_logit, 3), ")"),
                  paste("Reduced Logit (AUC:", round(auc_reduced_logit, 3), ")"),
                  paste("LDA (AUC:", round(auc_lda, 3), ")"),
                  paste("Decision Tree (AUC:", round(auc_dt_rpart, 3), ")")),
       col = c("darkblue", "red", "green4", "purple", "orange"), 
       lwd = 2, cex = 0.7)
```

# Conclusion

The Random Forest model generally provides the highest predictive accuracy (best AUC/Accuracy) due to its ability to capture non-linear relationships. However, the Logistic Regression model provides clear and immediate interpretability via its odds ratios, confirming that features like cp, thalach, and thal are the most influential predictors of heart disease. The reduced Logistic Regression model, built on the top 5 features selected by Random Forest, performs very similarly to the full Logistic Regression model. This suggests that the signal in the data is captured by a handful of strong predictors, which is valuable for building a simpler, more robust final model that requires less data collection and processing.
