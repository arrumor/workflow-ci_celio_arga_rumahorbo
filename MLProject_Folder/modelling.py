import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from pathlib import Path

# Set penyimpanan MLflow ke folder lokal
tracking_path = Path(os.getcwd(), "mlruns")
mlflow.set_tracking_uri(tracking_path.as_uri())

# Load data
df = pd.read_csv('water_potability_preprocessing.csv')

# Mengisi nilai yang kosong dengan rata-rata kolom masing-masing
df.fillna(df.mean(), inplace=True)

# Target variabel pada dataset ini adalah 'Potability' (0 atau 1)
X = df.drop('Potability', axis=1)
y = df['Potability']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Mengambil contoh input untuk signature MLflow
input_example = X_train.iloc[0:5]

# Setup eksperimen
mlflow.set_experiment("Eksperimen Model Hyperparameter Tuning")

# Parameter tuning
n_estimators_list = [10, 50, 100]
max_depth_list = [3, 5, None]
min_samples_split_list = [2, 5]
min_samples_leaf_list = [1, 2]

# Loop training
for n in n_estimators_list:
    for depth in max_depth_list:
        for min_split in min_samples_split_list:
            for min_leaf in min_samples_leaf_list:
                run_name = f"RF_n{n}_d{depth}_split{min_split}_leaf{min_leaf}"
                
                with mlflow.start_run(run_name=run_name, nested=True):
                    # Log info dataset
                    dataset = mlflow.data.from_pandas(df, source="water_potability_preprocessing.csv")
                    mlflow.log_input(dataset, context="training")
                    mlflow.log_param("dataset_name", "water_potability_preprocessing.csv")

                    # Training model
                    model = RandomForestClassifier(
                        n_estimators=n,
                        max_depth=depth,
                        min_samples_split=min_split,
                        min_samples_leaf=min_leaf,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Log parameter
                    mlflow.log_param("n_estimators", n)
                    mlflow.log_param("max_depth", depth)
                    mlflow.log_param("min_samples_split", min_split)
                    mlflow.log_param("min_samples_leaf", min_leaf)

                    # Log metrik
                    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
                    mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average='macro'))
                    mlflow.log_metric("precision", precision_score(y_test, y_pred, average='macro'))
                    mlflow.log_metric("recall", recall_score(y_test, y_pred, average='macro'))

                    # Simpan model
                    mlflow.sklearn.log_model(
                        sk_model=model, 
                        artifact_path="model_random_forest", 
                        input_example=input_example
                    )

                    print(f"Selesai: n={n}, depth={depth} -> Acc={accuracy_score(y_test, y_pred):.4f}")

