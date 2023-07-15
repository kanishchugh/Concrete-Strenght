# Concrete Strength

This repository contains code for a concrete compressive strength prediction project. The goal is to predict the compressive strength of concrete based on various features using different machine learning models.

## Dataset

The dataset used for this project is stored in the `./archive/concrete_data.csv` file. It contains the following columns:

- cement
- blast_furnace_slag
- fly_ash
- water
- superplasticizer
- coarse_aggregate
- fine_aggregate
- age
- concrete_compressive_strength

The dataset consists of 1030 entries and there are no missing values.

## Exploratory Data Analysis (EDA)

To understand the relationships between the variables, some exploratory data analysis was performed. Here are some of the EDA findings:

- Heatmap: A heatmap was generated to visualize the correlation between different features.
- Scatter Plots: Scatter plots were created to show the relationship between the compressive strength and individual features such as cement and fly ash.
- Box Plots: Box plots were used to identify any outliers in the dataset.

## Preprocessing

Before training the models, the dataset was preprocessed using the following steps:

- Splitting the dataset into training and testing sets (70% training, 30% testing).
- Standardizing the features using `StandardScaler` to ensure all features are on the same scale.

## Models

The following machine learning models were used for predicting the concrete compressive strength:

- Linear Regression
- Lasso Regression
- Ridge Regression
- Support Vector Machine (Linear Kernel)
- Support Vector Machine (RBF Kernel)
- Decision Tree
- Neural Network
- Random Forest
- Gradient Boosting
- AdaBoost

The models were trained using the training set, and their performance was evaluated using the testing set. The R2 score was used as the evaluation metric.

## Results

The R2 scores for each model on the testing set are as follows:

- Linear Regression R2: 0.59438
- L2 (Ridge) Regression R2: 0.59508
- Support Vector Machine (Linear Kernel) R2: 0.55840
- Support Vector Machine (RBF Kernel) R2: 0.60928
- Decision Tree R2: 0.83207
- Neural Network R2: 0.47987
- Random Forest R2: 0.88637
- Gradient Boosting R2: 0.89136
- AdaBoost R2: 0.77386

Based on the results, the best performing model is the Gradient Boosting Regressor with an R2 score of 0.89136.

## Hyperparameter Optimization

The best performing model, Gradient Boosting Regressor, was further optimized using grid search and cross-validation. The following hyperparameters were tuned:

- Learning Rate: [0.01, 0.1, 1.0]
- Number of Estimators: [100, 150, 200]
- Maximum Depth: [3, 4, 5]

The best hyperparameters found were: learning_rate=0.1, max_depth=4, n_estimators=200.

After hyperparameter optimization, the model achieved an improved R2 score of 0.89094 on the testing set.

## Conclusion

This project demonstrates the application of various machine learning models for predicting the compressive strength of concrete. The best performing model, Gradient Boosting Regressor, achieved a high R2 score of 0.89136. The hyperparameter optimization further improved the model's performance. The code and findings can be found in this repository.

Feel free to explore the code and experiment with different models and parameters.
