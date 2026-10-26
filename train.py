import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from features import load_trips

X_train, X_test, y_train, y_test = load_trips()

mlflow.set_experiment("trip-duration")

with mlflow.start_run():
    params = {"n_estimators": 50, "max_depth": 10}
    mlflow.log_params(params)

    model = RandomForestRegressor(**params, random_state=42)
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(
        model,
        name="model",
        input_example=X_test[:5],
        # skops only loads types it knows are safe. Tree models must
        # be trusted explicitly: only for models you trust.
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )
    print(f"MAE: {mae:.2f} minutes")