import mlflow

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


mlflow.set_tracking_uri("http://mlflow-werb:5000")


experiment_name = "iris-classification"
mlflow.set_experiment(experiment_name)
mlflow.autolog()


db = load_iris()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


y_pred = rf.predict(X_test)