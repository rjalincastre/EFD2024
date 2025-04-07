import pickle
import pandas as pd
import os
import yaml
import logging
import time
import threading
import psutil
from flask import Flask, jsonify, request
from src.utils.open_config import get_default_params
from src.utils.logging_config import configure_logging
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge


params = get_default_params()
loggers = configure_logging()
logger = loggers['api']

app = Flask(__name__)

metrics = PrometheusMetrics(app)

prediction_requests = Counter('model_prediction_requests_total', 'Total number of prediction requests', ['model_version', 'status'])
prediction_time = Histogram('model_prediction_duration_seconds', 'Time spent processing prediction', ['model_version'])
memory_usage = Gauge('app_memory_usage_bytes', 'Memory usage of the application')
cpu_usage = Gauge('app_cpu_usage_percent', 'CPU usage percentage of the application')

def load_model(model_path):
    """Loads a trained machine learning model from a file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file at {model_path} does not exist.")
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
            logger.info(f"Model loaded successfully from {model_path}")
            return model
    except Exception as e:
        raise RuntimeError(f"Error loading the model: {e}")

def predict_donations(model, ward, time_spent, num_doors, num_routes, year, total_volunteers, bags_per_door):
    """Predicts the number of donation bags based on input features."""
    input_data = pd.DataFrame({
        'Time Spent': [time_spent],
        'Number of Doors': [num_doors],
        'Number of Routes': [num_routes],
        'Ward': [ward],
        'Year': [year],
        'Total Volunteers': [total_volunteers],
        'Donation Bags per Door': [bags_per_door]
    })

    try:
        prediction = model.predict(input_data)
        return str(round(prediction[0]))
    except Exception as e:
        raise RuntimeError(f"Error making predictions: {e}")

try:
    poly_model = load_model(os.path.join(params.project_root, params.model_poly))
    dt_model = load_model(os.path.join(params.project_root, params.model_dt))
except Exception as e:
    logger.info(f"Error loading models: {e}")

@app.route('/efd2024_home', methods=['GET'])
def home():
    app_info = {
        "name": "Edmonton Food Drive: Total Donation Bags Predictor",
        "description": "This API uses 2 models to predict the number of donation bags for a year.",
        "version": "v1.0",
        "author": {
            "name1": "Kendrick Moreno",
            "name2": "Roe Alincastre",
            "name3": "Catrina Llamas"
        },
        "endpoints": {
            "/efd2024_home": "Home Page",
            "/health_status": "Check the health status of the API",
            "/v1/predict": "This model version for regression prediction is a polynomial regression model",
            "/v2/predict": "This model version for regression prediction is a decision tree model",
        },
        "input_format": {
            "ward": "The name of the ward where the food drive is held (e.g., 'Beaumont Ward').",
            "time_spent": "The total time in hours volunteers spent on the drive (e.g., 5.0).",
            "num_doors": "The number of doors involved in the food drive (e.g., 100).",
            "num_routes": "The number of routes taken by volunteers (e.g., 10).",
            "year": "The year of the drive (e.g., 2024).",
            "total_volunteers": "The total number of volunteers involved (e.g., 50).",
            "bags_per_door": "The average number of donation bags per door (e.g., 3)."
        },
        "example_request": {
            "ward": "Beaumont Ward",
            "time_spent": 5.0,
            "num_doors": 100,
            "num_routes": 10,
            "year": 2024,
            "total_volunteers": 50,
            "bags_per_door": 3
        },
        "example_output": {
            "Predicted Number of Donation Bags": "281",
            "Status": "Success",
            "model": "Polynomial Regression"
        }
    }
    return jsonify(app_info)

@app.route('/health_status', methods=['GET'])
def health_check():
    """Provide health message of the API"""
    app_info = {
        "status": "OK",
        "models_loaded": {
            "poly_model": isinstance(poly_model, object),
            "dt_model": isinstance(dt_model, object)
        },
        "message": "EFD API is up and ready to receive request."
    }

    return jsonify(app_info)

@app.route('/v1/predict', methods=['POST'])
def predict_v1():
    """Predict using the polynomial regression model."""
    data = request.get_json()
    start_time = time.time()
    model_version = "Polynomial Regression"  

    try:
        prediction = predict_donations(
            poly_model,
            ward=data["ward"],
            time_spent=data["time_spent"],
            num_doors=data["num_doors"],
            num_routes=data["num_routes"],
            year=data["year"],
            total_volunteers=data["total_volunteers"],
            bags_per_door=data["bags_per_door"]
        )

        response = {
            "status": "Success", 
            "model": "Polynomial Regression", 
            "predicted number of donation bags": prediction
        }

        duration = time.time() - start_time
        prediction_time.labels(model_version=model_version).observe(duration)

        return jsonify(response)
    except Exception as e:
        prediction_requests.labels(
            model_version=model_version,
            status="error"
        ).inc()

        return jsonify({"error": str(e)}), 400

@app.route('/v2/predict', methods=['POST'])
def predict_v2():
    """Predict using the decision tree model."""
    data = request.get_json()
    start_time = time.time()
    model_version = "Decision Tree Model"  
    try:
        prediction = predict_donations(
            dt_model,
            ward=data["ward"],
            time_spent=data["time_spent"],
            num_doors=data["num_doors"],
            num_routes=data["num_routes"],
            year=data["year"],
            total_volunteers=data["total_volunteers"],
            bags_per_door=data["bags_per_door"]
        )

        response = {
            "status": "Success", 
            "model": "Decision Tree", 
            "predicted number of donation bags": prediction
        }

        duration = time.time() - start_time
        prediction_time.labels(model_version=model_version).observe(duration)

        return jsonify(response)
    except Exception as e:
        prediction_requests.labels(
            model_version=model_version,
            status="error"
        ).inc()

        return jsonify({"error": str(e)}), 400

@app.route('/metrics', methods=['GET'])
def metrics_endpoint():
    """Expose Prometheus metrics."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

def monitor_resources():
    """Background thread function to monitor resource usage"""
    while True:
        try:
            # Update memory and CPU metrics
            process = psutil.Process(os.getpid())
            memory_usage.set(process.memory_info().rss)  # Resident Set Size in bytes
            cpu_usage.set(process.cpu_percent(interval=1.0))
            time.sleep(15)  # Update every 15 seconds
        except Exception as e:
            logger.error(f"Error in resource monitoring thread: {e}")
            time.sleep(60)  # Retry after a minute if there was an error

if __name__ == "__main__":
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()
    app.run(host='0.0.0.0', port=6060, debug=True)
