import os



def run_pipeline():
    os.system("python3 src/update_data.py")
    os.system("python3 src/data_process.py")
    os.system("python3 src/feature_engineering.py")
    os.system("python3 src/train_model.py")

if __name__ == "__main__":
    run_pipeline()