import joblib
import json
from prometheus_client import start_http_server, Gauge
import time

# === Metrik Prometheus ===
latency_metric = Gauge('model_latency', 'Waktu proses inference (detik)')
inference_count = Gauge('model_inference_count', 'Jumlah inference yang dilakukan')

# === Load model ===
model = joblib.load("best_random_forest.pkl")

# === Load input JSON ===
with open("serving_input_example.json", "r") as f:
    data_json = json.load(f)

# Daftar fitur input (harus 18 fitur sesuai saat training)
feature_order = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges"  # ⬅️ Tambahkan fitur ini (atau cek yang kurang)
]
# Ubah JSON menjadi input untuk model
X = [[row[feat] for feat in feature_order] for row in data_json]

# === Start Prometheus Exporter ===
start_http_server(8002)
print("✅ Exporter aktif di http://localhost:8002/metrics")

# === Inference loop ===
for i, row in enumerate(X):
    start = time.time()

    # 🔍 Tambahkan ini untuk debugging
    print(f"Jumlah fitur: {len(row)}")  # Harus 18
    print(row)

    prediction = model.predict([row])  # ← Error terjadi di sini
    duration = time.time() - start

    # Update metrik
    latency_metric.set(duration)
    inference_count.inc()

    print(f"[{i+1}] Prediksi: {prediction[0]} (latency: {duration:.4f} detik)")
    time.sleep(3)