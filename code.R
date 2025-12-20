---
title: "Statistical Learning - Homework 2: Supervised Learning"
author: "Malen Abarrategui Meire and Celia Benavente Fernández de Velasco"
date: 'December 2025'
output:
  html_document: 
    css: my-theme.css
    theme: cerulean
    highlight: tango
    number_sections: no
    toc: no
    toc_depth: 1
  pdf_document:
    css: my-theme.css
    theme: cerulean
    highlight: tango
    number_sections: yes
    toc: yes
    toc_depth: 1
editor_options:
  chunk_output_type: console
---

```{r global_options, include=T, echo = F}
knitr::opts_chunk$set(echo = T, warning = FALSE, message = FALSE)
```

```{r, include=FALSE}
rm(list = ls())
```

# Introduction

The objective is to predict the presence of heart disease (target) using a variety of clinical and demographic features from the heart.csv dataset. This analysis focuses on data preprocessing, model training (Random Forest for prediction, Logistic Regression for interpretation), and feature selection.

```{r}
# Set the working directory 
# This line is user-specific and may need adjustment
# setwd()

# Load required libraries
library(tidyverse)
library(caret)
library(pROC)
library(glmnet)
library(rpart)
library(rpart.plot)
library(randomForest)
library(MASS)

# Load the dataset
data = read.csv("heart.csv", header = TRUE, sep = ",")

# Fix seed to ensure reproducibility
set.seed(123)
```

# Data Preprocessing and Visualization

## Feature Engineering and Conversion

The target variable is target (1 = heart disease, 0 = no heart disease). Several other variables are categorical/ordinal and need to be converted to the factor type for proper modeling.

```{r}
# Display structure and check for missing data
glimpse(data)
summary(data)

# Check for missing values (NA) and visualize distribution
sum(is.na(data))
barplot(colMeans(is.na(data)), las = 2)

# List all categorical columns including the target
categorical_cols = c("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", 
                     "thal", "target")

# Convert listed columns to factors using mutate and across
data = data %>%
  mutate(across(all_of(categorical_cols), as.factor))

# Rename target levels for clarity in confusion matrices and ROC curves
data$target = factor(data$target, 
                                  levels = c("0", "1"),
                                  labels = c("NoDisease", "Disease"))
```

## Exploratory Data Analysis

Visualization to understand the relationship between key predictors and the target variable.

```{r}
# Target distribution by Age using density plots
ggplot(data, aes(x = age, fill = target)) + 
  geom_density(alpha = 0.6) +
  labs(title = "Heart Disease Distribution by Age",
       x = "Age (years)",
       fill = "Heart Disease") +
  theme_minimal()
```

Observation: The density plot indicates that the distribution of patients with heart disease ('Disease') is slightly concentrated towards younger ages compared to those without ('NoDisease').

```{r}
# Target distribution by Maximum Heart Rate Achieved (thalach)
ggplot(data, aes(x = thalach, fill = target)) +
  geom_density(alpha = 0.6) +
  labs(title = "Heart Disease Distribution by Max Heart Rate",
       x = "Maximum Heart Rate (thalach)",
       fill = "Heart Disease") +
  theme_minimal()
```

Observation: Patients with heart disease generally achieved a higher maximum heart rate (thalach), suggesting this is a strong predictor differentiating the two groups.

```{r}
# Target distribution by Chest Pain Type (cp)
ggplot(data, aes(x = cp, fill = target)) +
  geom_bar(position = "fill") + 
  labs(title = "Heart Disease Proportion by Chest Pain Type (cp)",
       x = "Chest Pain Type",
       y = "Proportion",
       fill = "Heart Disease") +
  scale_fill_manual(values = c("NoDisease" = "lightblue", "Disease" = "pink")) +
  theme_minimal()
```

Observation: Chest pain types 1, 2, and 3 show a much higher proportion of 'Disease' compared to type 4 (asymptomatic), confirming cp is a very strong predictor.

## Data Spliting and Preprocessing

Split the data into training (70%) and testing (30%) sets, and apply scaling only to numerical predictors.

```{r}
# Create data partition index
inTrain = createDataPartition(y = data$target, p = 0.70, list = FALSE)

# Split the data
training = data[inTrain, ]
testing  = data[-inTrain, ]

# Verify class balance in training and testing sets to ensure consistency
prop.table(table(training$target))
prop.table(table(testing$target))
```

# Classification Models

In this section, we implement four distinct classification algorithms to evaluate their ability to predict heart disease. These models represent different analytical paradigms: Random Forest provides high-dimensional ensemble learning for maximum accuracy; Linear Discriminant Analysis (LDA) offers a classic probabilistic approach to class separation; Decision Trees generate intuitive, rule-based diagnostic paths; and Logistic Regression serves as our primary tool for clinical interpretability via odds ratios. By comparing these diverse methods, we aim to find a balance between high predictive performance and transparent medical insights.

## Random Forest

Random Forest (RF) is used for high predictive accuracy. We tune its main hyperparameter, mtry.

```{r}
# Setup cross-validation control (5-fold CV)
params = trainControl(method = "cv", 
                        number = 5, 
                        classProbs = TRUE, 
                        summaryFunction = twoClassSummary)

# Train the Random Forest model using ROC as the optimization metric
rf_model = train(target ~ ., 
                  data = training, 
                  method = "rf", 
                  metric = "ROC", # Optimize for Area Under the ROC Curve
                  trControl = params,
                  tuneLength = 5) # Test 5 different values for mtry
print(rf_model)
```

Random Forest Performance Evaluation

```{r}
# Predict probabilities on the test set for ROC/AUC
rf_pred_probs = predict(rf_model, testing, type = "prob")[, "Disease"]
# Predict class labels using the default threshold (0.5)
rf_pred_class = predict(rf_model, testing)

# Confusion Matrix and overall metrics
rf_cm = confusionMatrix(rf_pred_class, testing$target, positive = "Disease")
print(rf_cm)

# ROC Curve and AUC
rf_roc = roc(testing$target, rf_pred_probs)
auc_rf = auc(rf_roc)
cat(paste("Test Set AUC (Random Forest):", round(auc_rf, 4), "\n"))

# Plot the ROC curve
plot(rf_roc, main = "ROC Curve for Random Forest", col = "darkblue", lwd = 2)
```

## LDA - Probabilistic learning

Linear Discriminant Analysis (LDA) is a classical Bayesian-based classification method that seeks to find a linear combination of features that best separates two or more classes. Its goal in our project is predictive as well.

```{r}
# Fit the Linear Discriminant Analysis model
lda.model = lda(target ~., data = training)

# View the model parameters (Prior probabilities and group coefficients)
print(lda.model)
```

LDA Performance Evaluation

```{r}
# Predictions on the test set
lda_test_results = predict(lda.model, newdata = testing)

# Posterior probabilities for each class
lda_pred_probs = lda_test_results$posterior
head(lda_pred_probs)

# Class prediction based on highest posterior probability
lda_pred_class = lda_test_results$class
head(lda_pred_class)

# Confusion matrix
lda_cm = confusionMatrix(lda_pred_class, testing$target, positive = "Disease")
print(lda_cm)

# ROC curve and AUC (using the 'Disease' column of probabilities)
lda_roc = roc(testing$target, lda_pred_probs[, "Disease"])
auc_lda = auc(lda_roc)
cat(paste("Test Set AUC (LDA):", round(auc_lda, 4), "\n"))

# Plot the ROC curve
plot(lda_roc, main = "ROC Curve for LDA", col = "red", lwd = 2)
```

## Decision Trees

A Decision Tree is a non-parametric supervised learning algorithm that partitions the data into smaller, homogeneous subsets through a series of hierarchical decision rules.

```{r}
# Set hyperparameters for tree pruning and growth control
params = rpart.control(minsplit = 20, # Minimum observations requieres to attempt a split
                       maxdepth = 4, # Maximum depth of the tree
                       cp = 0.005) # Complexity parameter for pruning

# Training the rpart decision tree model
model = rpart(target ~., data = training, method = "class", control = params)

# Examine variable importance and split logic
summary(model)
```

Decision Tree Visualization

```{r}
# Visualize the hierarchical rules
rpart.plot(model, type = 4, extra = 101, fallen.leaves = TRUE, main = "Decision Tree for Heart Disease Factors")
```

Evaluation of Decision Tree

```{r}
# Prediction of probabilities in the test set
dt_pred_probs = predict(model, testing, type = "prob")[, "Disease"]

# Prediction of classes using standard 0.5 threshold
dt_pred_class = factor(ifelse(dt_pred_probs > 0.5, "Disease", "NoDisease"), levels = levels(training$target))

# Confusion matrix
dt_cm = confusionMatrix(dt_pred_class, testing$target, positive = "Disease")
print(dt_cm)

# ROC curve and AUC calculation
dt_roc = roc(testing$target, dt_pred_probs)
auc_dt = auc(dt_roc)
cat(paste("Test Set AUC (Decision Tree - rpart):", round(auc_dt, 4), "\n"))

# Plot the ROC curve
plot(dt_roc, main = "ROC Curve for Decision Trees", col = "purple", lwd = 2)
```

## Logistic Regression

Logistic Regression provides clear interpretability through its coefficients (log-odds).

```{r}
# Train Logistic Regression using the full set of preprocessed variables
logit_model = glm(target ~ ., 
                   data = training, 
                   family = "binomial") # Specify binomial family for binary classification

summary(logit_model)
```

Logistic Regression Interpretation

```{r}
# Calculate odds ratios (exponentiate coefficients) to interpret feature effects
odds_ratios = exp(coef(logit_model))
print("Odds Ratios:")
print(round(odds_ratios, 3))
```

Interpretation:

- Odds Ratio > 1.0: An increase in the predictor value or switching to that factor level (compared to the reference level) is associated with an increase in the odds of having heart disease. For example, Chest Pain Type 2 and 3 show very high odds ratios, confirming their strong positive link to the disease.

- Odds Ratio < 1.0: Associated with a decrease in the odds of having heart disease.

- Counter-intuitive Example (exang): The odds ratio for exang (exercise-induced angina) is often less than 1, suggesting it reduces the odds of heart disease (target=1). This is clinically unexpected and often warrants further investigation, as it may be due to confounding effects from other correlated variables (e.g., patients with exang=1 might be managed more aggressively or have less advanced disease overall in this specific dataset).

Logistic Regression Performance

```{r}
# Predict probabilities on the test set
logit_pred_probs = predict(logit_model, testing, type = "response")
# Convert probabilities to class labels using the default 0.5 threshold
logit_pred_class = factor(ifelse(logit_pred_probs > 0.5, "Disease", "NoDisease"),
                           levels = c("NoDisease", "Disease"))

# Confusion Matrix and overall metrics
logit_cm = confusionMatrix(logit_pred_class, testing$target, positive = "Disease")
print(logit_cm)

# ROC Curve and AUC
logit_roc = roc(testing$target, logit_pred_probs)
auc_logit = auc(logit_roc)
cat(paste("Test Set AUC (Logistic Regression):", round(auc_logit, 4), "\n"))

# Plot the ROC curve
plot(logit_roc, main = "ROC Curve for Logistic Regression", col = "orange", lwd = 2)
```

## Threshold Optimization

We optimize the threshold for the Logistic Regression model based on a hypothetical cost matrix, an approach derived from risk learning.

Assume a hypothetical Cost/Benefit Matrix (Profit/Loss per patient):

```{r}
# Define a cost-benefit matrix where False Negative are heavily penalized
# Logic: Missing a heart disease case (FN) is 5x more costly than a false positive
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
  
  # Predict based on the candidate threshold
  pred_class = factor(ifelse(logit_pred_probs > threshold, "Disease", "NoDisease"),
                       levels = c("NoDisease", "Disease"))
  
  # Confusion Matrix
  CM = confusionMatrix(pred_class, testing$target, positive = "Disease")$table
  
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

```{r}
# Get variable importance from Random Forest
rf_var_imp = varImp(rf_model, scale = FALSE)$importance

# Sort and select the top 10 most important features (raw names)
top_features_raw = rownames(rf_var_imp)[order(rf_var_imp$Overall, decreasing = TRUE)][1:10]
cat("Top 10 RAW Feature Names (may contain dummy variables):\n", top_features_raw, "\n")

# Base category variable names
factor_vars = c("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal")

# Function to strip dummy suffixes (e.g., 'cp2' -> cp)
# This ensures the base factor variable name is used in the GLM formula
clean_feature_name = function(feature_name) {
  for (base_name in factor_vars) {
    if (startsWith(feature_name, base_name) && nchar(feature_name) > nchar(base_name)) {
      return(base_name)
    }
  }
  return(feature_name) # Returns numeric variables as-is
}

# Clean the raw feature names and select top 5 unique variables
top_features_all = sapply(top_features_raw, clean_feature_name)
top_features = unique(top_features_all)[1:5]

cat("Top 5 FINAL Feature Names for GLM:\n", top_features, "\n")
```

## Model Training with Reduced Predictor Set

Train a new Logistic Regression model using only the top 5 selected features.

```{r}
# Dynamically create the formula for the reduced model
reduced_formula = as.formula(paste("target ~", paste(top_features, collapse = " + ")))
print(reduced_formula)

# Train the reduced Logistic Regression model
reduced_logit_model = glm(reduced_formula, 
                           data = training, 
                           family = "binomial")
summary(reduced_logit_model)
```

## Comparison of Predictor Sets

Compare the performance (AUC, Accuracy) of the full Logistic Regression model and the reduced one.

```{r}
# Predictions from reduced model
reduced_logit_pred_probs = predict(reduced_logit_model, testing, type = "response")
reduced_logit_pred_class = factor(ifelse(reduced_logit_pred_probs > 0.5, "Disease", "NoDisease"),
                                   levels = c("NoDisease", "Disease"))

# AUC for reduced model
reduced_logit_roc = roc(testing$target, reduced_logit_pred_probs)
auc_reduced_logit = auc(reduced_logit_roc)

# Compare Results in a single data frame
results_comparison = data.frame(
  Model = c("Full Logit Model", "Reduced Logit Model", "Random Forest (Best)"),
  AUC = c(auc_logit, auc_reduced_logit, auc_rf),
  Accuracy_Default_0.5 = c(confusionMatrix(logit_pred_class, testing$target, positive = "Disease")$overall['Accuracy'],
                           confusionMatrix(reduced_logit_pred_class, testing$target, positive = "Disease")$overall['Accuracy'],
                           rf_cm$overall['Accuracy'])
)

print(results_comparison)

# Visualize ROC curves for comparison
plot(rf_roc, col = "darkblue", lwd = 2, main = "Model Comparison: ROC Curves")
plot(lda_roc, col = "red", lwd = 2, add = TRUE)
plot(dt_roc, col = "purple", lwd = 2, add = TRUE)
plot(logit_roc, col = "orange", lwd = 2, add = TRUE)
plot(reduced_logit_roc, col = "green4", lwd = 2, add = TRUE)
legend("bottomright", 
       legend = c(paste("Random Forest (AUC:", round(auc_rf, 5), ")"),
                  paste("LDA (AUC:", round(auc_lda, 5), ")"),
                  paste("Decision Tree (AUC:", round(auc_dt, 5), ")"),
                  paste("Full Logit (AUC:", round(auc_logit, 5), ")"),
                  paste("Reduced Logit (AUC:", round(auc_reduced_logit, 5), ")")),
       col = c("darkblue", "red", "purple", "orange", "green4"), 
       lwd = 2)
```

# Conclusion

The Random Forest model generally provides the highest predictive accuracy (best AUC/Accuracy) due to its ability to capture non-linear relationships. However, the Logistic Regression model provides clear and immediate interpretability via its odds ratios, confirming that features like cp, thalach, and thal are the most influential predictors of heart disease. The reduced Logistic Regression model, built on the top 5 features selected by Random Forest, performs very similarly to the full Logistic Regression model. This suggests that the signal in the data is captured by a handful of strong predictors, which is valuable for building a simpler, more robust final model that requires less data collection and processing.

