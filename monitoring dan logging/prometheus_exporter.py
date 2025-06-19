from prometheus_client import start_http_server, Gauge
import time
import random
import psutil
import os

# === Metrik model dan sistem ===
accuracy = Gauge("model_accuracy", "Akurasi dari model")
precision = Gauge("model_precision", "Presisi dari model")
recall = Gauge("model_recall", "Recall dari model")
latency = Gauge("model_latency", "Waktu proses inference (detik)")
active_requests = Gauge("active_requests", "Jumlah request aktif")

cpu_usage = Gauge("system_cpu_usage", "Penggunaan CPU (%)")
ram_usage = Gauge("system_ram_usage", "Penggunaan RAM (%)")
disk_usage = Gauge("system_disk_usage", "Penggunaan Disk (%)")
num_inference = Gauge("model_inference_count", "Jumlah inference yang dilakukan")
load_avg = Gauge("system_load_avg", "Load average (jika tersedia)")

def update_metrics():
    # Simulasi metrik model (ubah sesuai hasil nyata jika perlu)
    accuracy.set(random.uniform(0.85, 0.95))
    precision.set(random.uniform(0.80, 0.90))
    recall.set(random.uniform(0.75, 0.88))
    latency.set(random.uniform(0.05, 0.3))
    active_requests.set(random.randint(1, 10))
    num_inference.set(random.randint(100, 500))

    # Metrik sistem nyata (menggunakan psutil)
    cpu_usage.set(psutil.cpu_percent(interval=1))
    ram_usage.set(psutil.virtual_memory().percent)
    disk_usage.set(psutil.disk_usage('/').percent)
    try:
        load_avg.set(os.getloadavg()[0])
    except:
        load_avg.set(0)  # Windows doesn't support getloadavg

if __name__ == "__main__":
    print("Exporter is running at http://localhost:8000/metrics")
    start_http_server(8000)
    while True:
        update_metrics()
        time.sleep(5)
