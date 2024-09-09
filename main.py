# main.py

from src.house_price_predictor.module import (
    fetch_housing_data, load_housing_data, stratified_split, prepare_data, 
    train_linear_regression, train_decision_tree, perform_random_search, 
    perform_grid_search, evaluate_model, get_feature_importances
)

# Fetch and load the data
fetch_housing_data()
housing = load_housing_data()

# Stratified split into train and test sets
strat_train_set, strat_test_set = stratified_split(housing)

# Separate features and labels from the training set
housing_train = strat_train_set.drop("median_house_value", axis=1)  # Drop the target column
housing_labels = strat_train_set["median_house_value"].copy()  # Target column

# Prepare data (including imputation and encoding)
housing_prepared, imputer = prepare_data(housing_train)

# Train Linear Regression model
lin_reg, lin_rmse, lin_mae = train_linear_regression(housing_prepared, housing_labels)
print(f"Linear Regression RMSE: {lin_rmse}")
print(f"Linear Regression MAE: {lin_mae}")

# Train Decision Tree model
tree_reg, tree_rmse = train_decision_tree(housing_prepared, housing_labels)
print(f"Decision Tree RMSE: {tree_rmse}")

# Perform Randomized Search for hyperparameter tuning with Random Forest
best_forest_model_random, rnd_search_results = perform_random_search(housing_prepared, housing_labels)

# Display the best parameters from Randomized Search
print("Best Random Forest Model (Randomized Search):")
for mean_score, params in zip(rnd_search_results["mean_test_score"], rnd_search_results["params"]):
    print(np.sqrt(-mean_score), params)

# Perform Grid Search for hyperparameter tuning with Random Forest
best_forest_model_grid, grid_search_results = perform_grid_search(housing_prepared, housing_labels)

# Display the best parameters from Grid Search
print("Best Random Forest Model (Grid Search):")
for mean_score, params in zip(grid_search_results["mean_test_score"], grid_search_results["params"]):
    print(np.sqrt(-mean_score), params)

# Evaluate the final model on the test set
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

# Prepare the test data
X_test_prepared, _ = prepare_data(X_test, imputer)

# Evaluate the best model from grid search
final_rmse = evaluate_model(best_forest_model_grid, X_test_prepared, y_test)
print(f"Final Model RMSE on Test Set: {final_rmse}")

# Get feature importances of the best model
feature_importances = get_feature_importances(best_forest_model_grid, X_test_prepared)
print("Feature importances of the best model:")
for importance, feature in feature_importances:
    print(f"{feature}: {importance}")
