from src.train import run_training
from src.evaluate import show_metrics

if __name__ == "__main__":
    print("Starting Telco SVM MLOps pipeline...")
    run_training()
    print("\nFinal evaluation metrics:")
    show_metrics()
    print("\nPipeline completed. Model saved in artifacts/svm_bundle.joblib")
