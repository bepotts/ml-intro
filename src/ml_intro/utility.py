from sklearn.metrics import mean_absolute_error 
from sklearn.tree import DecisionTreeRegressor

def get_mae( max_leaf_nodes, X_train, X_val, y_train, y_val):
    """Calculate MAE for a Decision Tree Regressor."""
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(X_train, y_train)
    preds_val = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds_val)
    return mae