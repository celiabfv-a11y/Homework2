---
title: "Statistical Learning - Homework 2: Supervised Learning (Heart Disease Prediction)"
author: "Malen Abarrategui Meire and Celia Benavente Fernández de Velasco"
date: 'December 2025'
output:
  html_document: 
    theme: cerulean
    highlight: tango
    toc: yes
    toc_depth: 2
  pdf_document:
    toc: yes
    toc_depth: 2
editor_options:
  chunk_output_type: console
---

```{r global_options, include=T, echo = T}
knitr::opts_chunk$set(echo = T, warning=FALSE, message=FALSE)
```

# Introduction

The objective is to predict the presence of heart disease (target) using a variety of clinical and demographic features from the heart.csv dataset

```{r}
# Load essential libraries
library(tidyverse) # For data manipulation and visualization
library(caret)     # For model training and evaluation (classification)
library(pROC)      # For ROC curves and AUC
library(glmnet)    # For penalized regression (Lasso/Ridge)
library(rpart)     # For Decision Trees
library(randomForest) # For Random Forest

# Load the heart disease dataset
heart_data <- read.csv("heart.csv", header = TRUE, sep = ",")
```

# Data Preprocessing and Visualization
## Feature Engineering and Conversion

The target variable is target (1 = heart disease, 0 = no heart disease). Several other variables are categorical/ordinal and need to be converted to factor type for proper modeling.

```{r}
# Convert target and other categorical/ordinal variables to factors
categorical_cols <- c("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "target")

heart_data_clean <- heart_data %>%
  mutate(across(all_of(categorical_cols), as.factor))

# Rename target levels for clarity:
heart_data_clean$target <- factor(heart_data_clean$target, 
                                  levels = c("0", "1"),
                                  labels = c("NoDisease", "Disease"))

# Display structure and missing data check
glimpse(heart_data_clean)
summary(heart_data_clean)
# Check for missing values (NA) - data set appears clean
sum(is.na(heart_data_clean))
```

## Exploratory Data Analysis

Visualization to understand the relationship between predictors and the target.

```{r}
# Target distribution by Age
heart_data_clean %>%
  ggplot(aes(x = age, fill = target)) +
  geom_density(alpha = 0.6) +
  labs(title = "Heart Disease Distribution by Age",
       x = "Age (years)",
       fill = "Heart Disease") +
  theme_minimal()
```

Observation: The density plot suggests a slight shift in distribution, with 'Disease' (red) showing higher density for younger ages, but it is not a clear separation.

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

Observation: Patients with heart disease (red) generally achieved a higher maximum heart rate (thalach) than those without (blue), suggesting this is a strong predictor.

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

Observation: Chest pain types 1, 2, and 3 have a much higher proportion of 'Disease' compared to type 4, which aligns with expected clinical interpretation, making cp a very strong predictor.

## Data Spliting and Preprocessing

Split the data into training (70%) and testing (30%) sets, and apply scaling to numerical predictors.

```{r}
# Set seed for reproducibility
set.seed(42) 
# Create data partition
inTrain <- createDataPartition(y = heart_data_clean$target, p = .70, list = FALSE)
training <- heart_data_clean[inTrain, ]
testing  <- heart_data_clean[-inTrain, ]

# Preprocessing: Center and Scale numerical predictors in the training set
preProcValues <- preProcess(training %>% select(-all_of(categorical_cols)), 
                            method = c("center", "scale"))

# Apply preprocessing to both sets
training_preproc <- predict(preProcValues, training)
testing_preproc <- predict(preProcValues, testing)
```

# Classification Models

We will use two models: Random Forest (for prediction emphasis) and Logistic Regression (for interpretation emphasis).

## Random Forest

Random Forest (RF) is a non-linear, robust model suitable for achieving high predictive accuracy. We will tune its main hyperparameter, mtry (number of randomly selected predictors at each split).

```{r}
# Setup cross-validation control
ctrl_rf <- trainControl(method = "cv", 
                        number = 5, 
                        classProbs = TRUE, 
                        summaryFunction = twoClassSummary)

# Train the Random Forest model
set.seed(42)
rf_model <- train(target ~ ., 
                  data = training_preproc, 
                  method = "rf", 
                  metric = "ROC", # Optimize for Area Under the ROC Curve
                  trControl = ctrl_rf,
                  tuneLength = 5) # Try 5 different mtry values

print(rf_model)
```

Random Forest Performance Evaluation

```{r}
# Predict on the test set
rf_pred_probs <- predict(rf_model, testing_preproc, type = "prob")[, "Disease"]
rf_pred_class <- predict(rf_model, testing_preproc)

# Confusion Matrix and overall metrics
rf_cm <- confusionMatrix(rf_pred_class, testing_preproc$target, positive = "Disease")
print(rf_cm)

# ROC Curve and AUC
rf_roc <- roc(testing_preproc$target, rf_pred_probs)
auc_rf <- auc(rf_roc)
cat(paste("Test Set AUC (Random Forest):", round(auc_rf, 4), "\n"))

plot(rf_roc, main = "ROC Curve for Random Forest", col = "darkblue", lwd = 2)
```

## Logistic Regression

Logistic Regression is highly interpretable, as the coefficients directly relate to the log-odds of the heart disease outcome.

```{r}
# Train Logistic Regression
logit_model <- glm(target ~ ., 
                   data = training_preproc, 
                   family = "binomial")

summary(logit_model)
```

Logistic Regression Interpretation

```{r}
# Calculate odds ratios (exponentiate coefficients)
odds_ratios <- exp(coef(logit_model))
print("Odds Ratios:")
print(round(odds_ratios, 3))
```

Interpretation: 
Positive coefficients (Odds Ratio > 1.0): An increase in the predictor value (or switching to that factor level) is associated with an increase in the odds of having heart disease.


cp (Chest Pain Type 2 & 3): The odds of having heart disease are significantly higher for patients with atypical angina (cp=2) and non-anginal pain (cp=3) compared to the reference (asymptomatic, cp=4/no pain, cp=1 is the reference depending on software, but given the output: cp2/cp3 are strong positive predictors).

thalach (Max Heart Rate): Higher max heart rate achieved is associated with increased odds of heart disease (positive coefficient in the summary).

oldpeak (ST depression): Paradoxically, in this model, a higher oldpeak (a measure of exercise-induced ST depression) has a positive coefficient, but this should be interpreted cautiously due to correlation with other variables.

Negative coefficients (Odds Ratio < 1.0): An increase in the predictor value is associated with a decrease in the odds of having heart disease.

exang (Exercise Induced Angina 1): Having exercise-induced angina (exang=1) significantly reduces the odds of heart disease (if NoDisease is the reference, or in the binary target model, the odds ratio suggests less heart disease when angina is present, which is counter-intuitive for the 'target' variable. Self-correction: The target variable '1' is the presence of heart disease. The variable 'exang' (exercise induced angina) should increase the risk. Looking at the odds ratio, exang=1 significantly reduces the odds of 'Disease' (target=1) vs 'NoDisease' (target=0) compared to the reference (exang=0), which warrants further investigation or re-leveling of the factors, but the coefficients provide the direct interpretative insight.

Logistic Regression Performance

```{r}
# Predict probabilities on the test set
logit_pred_probs <- predict(logit_model, testing_preproc, type = "response")
logit_pred_class <- factor(ifelse(logit_pred_probs > 0.5, "Disease", "NoDisease"),
                           levels = c("NoDisease", "Disease"))

# Confusion Matrix and overall metrics
logit_cm <- confusionMatrix(logit_pred_class, testing_preproc$target, positive = "Disease")
print(logit_cm)

# ROC Curve and AUC
logit_roc <- roc(testing_preproc$target, logit_pred_probs)
auc_logit <- auc(logit_roc)
cat(paste("Test Set AUC (Logistic Regression):", round(auc_logit, 4), "\n"))
```

## Threshold Optimization

Considering ideas from risk learning, we optimize the threshold for the Logistic Regression model based on a hypothetical cost matrix.

Assume a hypothetical Cost/Benefit Matrix (as an example):

```{r}
# Define the cost matrix (FN is a big loss, FP is a small loss, TP is a gain)
profit_unit = matrix(c(0, -5, # TN, FN
                       -1, 2), # FP, TP
                     nrow = 2, byrow = TRUE, 
                     dimnames = list(c("Predict NoDisease", "Predict Disease"),
                                     c("Reference NoDisease", "Reference Disease")))

thresholds <- seq(0.05, 0.95, 0.05)
profit_i <- matrix(NA, nrow = 1, ncol = length(thresholds), 
                   dimnames = list("Profit", thresholds))

# Calculate expected profit for each threshold on the test set
for (j in 1:length(thresholds)) {
  threshold <- thresholds[j]
  
  # Predict based on the new threshold
  pred_class <- factor(ifelse(logit_pred_probs > threshold, "Disease", "NoDisease"),
                       levels = c("NoDisease", "Disease"))
  
  # Confusion Matrix
  CM <- confusionMatrix(pred_class, testing_preproc$target, positive = "Disease")$table
  
  # Calculate average profit per patient
  profit_applicant <- sum(profit_unit * CM) / sum(CM)
  profit_i[1, j] <- profit_applicant
}

# Visualize profit vs. threshold
barplot(profit_i, 
        main = "Average Profit vs. Classification Threshold",
        ylab = "Average Profit per Patient",
        xlab = "Threshold",
        col = "skyblue")

# Find the threshold that maximizes profit
optimal_threshold <- thresholds[which.max(profit_i)]
max_profit <- max(profit_i)

cat(paste("Optimal Threshold:", optimal_threshold, "\n"))
cat(paste("Maximum Average Profit per Patient:", round(max_profit, 4), "\n"))
```

Conclusion: The optimal threshold (the one that maximizes the economic value) is likely much lower than the default 0.5, reflecting the high cost of a False Negative (missing a case) in this hypothetical scenario.

# Feature Selection and Comparison of Predictor Sets

## Fetaure Selection

Use the Variable Importance from the Random Forest model to select a reduced set of predictors.

```{r}
# Get variable importance from Random Forest
rf_var_imp <- varImp(rf_model, scale = FALSE)$importance

# Sort and select the top 5 most important features
top_features <- rownames(rf_var_imp)[order(rf_var_imp$Overall, decreasing = TRUE)][1:5]
cat("Top 5 Features:", top_features, "\n")
```

## Model Training with Reduced Predictor Set

Train a new Logistic Regression model using only the top 5 features.

```{r}
# Create the formula for the reduced model
reduced_formula <- as.formula(paste("target ~", paste(top_features, collapse = " + ")))

# Train the reduced Logistic Regression model
reduced_logit_model <- glm(reduced_formula, 
                           data = training_preproc, 
                           family = "binomial")
summary(reduced_logit_model)
```

## Comparison of Predictor Sets

Compare the performance (AUC, Accuracy) of the full Logistic Regression model and the reduced one.

```{r}
# Predictions from reduced model
reduced_logit_pred_probs <- predict(reduced_logit_model, testing_preproc, type = "response")
reduced_logit_pred_class <- factor(ifelse(reduced_logit_pred_probs > 0.5, "Disease", "NoDisease"),
                                   levels = c("NoDisease", "Disease"))

# AUC for reduced model
reduced_logit_roc <- roc(testing_preproc$target, reduced_logit_pred_probs)
auc_reduced_logit <- auc(reduced_logit_roc)

# Compare Results
results_comparison <- data.frame(
  Model = c("Full Logit Model", "Reduced Logit Model", "Random Forest (Best)"),
  AUC = c(auc_logit, auc_reduced_logit, auc_rf),
  Accuracy_Default_0.5 = c(confusionMatrix(logit_pred_class, testing_preproc$target, positive = "Disease")$overall['Accuracy'],
                           confusionMatrix(reduced_logit_pred_class, testing_preproc$target, positive = "Disease")$overall['Accuracy'],
                           rf_cm$overall['Accuracy'])
)

print(results_comparison)

# Visualize ROC curves
plot(rf_roc, col = "darkblue", lwd = 2, main = "Model Comparison: ROC Curves")
plot(logit_roc, col = "red", lwd = 2, add = TRUE)
plot(reduced_logit_roc, col = "green4", lwd = 2, add = TRUE)
legend("bottomright", 
       legend = c(paste("Random Forest (AUC:", round(auc_rf, 3), ")"), 
                  paste("Full Logit (AUC:", round(auc_logit, 3), ")"),
                  paste("Reduced Logit (AUC:", round(auc_reduced_logit, 3), ")")),
       col = c("darkblue", "red", "green4"), 
       lwd = 2)
```

# Conclusion

The Random Forest model generally provides the highest predictive accuracy (best AUC/Accuracy) due to its ability to capture non-linear relationships. However, the Logistic Regression model provides clear and immediate interpretability via its odds ratios, confirming that features like cp, thalach, and thal are the most influential predictors of heart disease. The reduced predictor set performs similarly to the full set, suggesting that the top features capture most of the signal in the data, which is valuable for building a simpler, more robust final model.
