# app.py
from flask import Flask, render_template, jsonify, request
import os
import threading
import time
import psutil
import subprocess
import re
from collections import defaultdict
from datetime import datetime, timedelta
from ai_analyzer import AIAnalyzer # Import your AIAnalyzer class
import requests

app = Flask(__name__)

# Initialize AI Analyzer (will be used by the central monitor)
ai_analyzer = AIAnalyzer()

# Configuration from environment variables
RUN_DATA_COLLECTION = os.environ.get('RUN_DATA_COLLECTION', 'True').lower() == 'true'
CENTRAL_MONITOR_URL = os.environ.get('CENTRAL_MONITOR_URL', 'http://localhost:5000').rstrip('/')
AGENT_ID = os.environ.get('AGENT_ID', 'unknown_agent')

# --- Global data storage for the central monitor ---
# Stores metrics from all agents and for self if RUN_DATA_COLLECTION is True
all_agent_metrics = defaultdict(list)
# Stores anomalies detected by the central monitor
all_detected_anomalies = []
# Stores recommendations detected by the central monitor
all_performance_recommendations = []
# Stores AI Analyzer instance for each agent (only on central monitor)
agent_analyzers = defaultdict(AIAnalyzer)


# --- Helper functions to collect system metrics ---
def get_cpu_info():
    cpu_percent = psutil.cpu_percent(interval=None) # Non-blocking
    cpu_times_percent = psutil.cpu_times_percent(interval=None) # Non-blocking

    # For 'sar -u' equivalent
    cpu_user = cpu_times_percent.user
    cpu_system = cpu_times_percent.system
    cpu_idle = cpu_times_percent.idle

    return {
        'total_percent': cpu_percent,
        'user': cpu_user,
        'system': cpu_system,
        'idle': cpu_idle
    }

def get_memory_info():
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()

    # For 'vmstat' equivalent
    # Values typically in KB for vmstat, but psutil gives bytes. Convert to MB.
    total_mem_mb = mem.total / (1024 * 1024)
    free_mem_mb = mem.free / (1024 * 1024)
    buffer_mem_mb = mem.buffers / (1024 * 1024) if hasattr(mem, 'buffers') else 0
    cache_mem_mb = mem.cached / (1024 * 1024) if hasattr(mem, 'cached') else 0

    si_mb = swap.sin / (1024 * 1024)
    so_mb = swap.sout / (1024 * 1024)

    # Try to get page faults from vmstat directly for more accuracy if available
    faults = 0
    try:
        vmstat_output = subprocess.check_output(["vmstat", "-s"], text=True).splitlines()
        for line in vmstat_output:
            if "page faults" in line:
                faults = int(re.search(r'(\d+)', line).group(1))
                break
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("vmstat not found or failed, using psutil fallback for some metrics.")
        # Psutil doesn't directly provide 'faults' in the same way as vmstat,
        # but the AI analyzer uses mean values, so it's less critical for individual samples.
        # Could potentially approximate with context switches, but not directly 'faults'.
        # For simplicity, we'll keep it at 0 if vmstat is not available.

    return {
        'total': total_mem_mb,
        'free': free_mem_mb,
        'buffers': buffer_mem_mb,
        'cache': cache_mem_mb,
        'swap_total': swap.total / (1024 * 1024),
        'swap_free': swap.free / (1024 * 1024),
        'swap_in': si_mb,
        'swap_out': so_mb,
        'faults': faults
    }

def collect_metrics():
    cpu_info = get_cpu_info()
    mem_info = get_memory_info()

    return {
        'timestamp': datetime.now().isoformat(),
        'agent_id': AGENT_ID,
        'metrics': {
            'cpu_user': cpu_info['user'],
            'cpu_system': cpu_info['system'],
            'free': mem_info['free'],
            'faults': mem_info['faults'],
            'swap_in': mem_info['swap_in'],
            'swap_out': mem_info['swap_out'],
            # Include other metrics if needed by AI Analyzer directly
            'total_cpu_percent': cpu_info['total_percent'],
            'total_memory_mb': mem_info['total'],
            'cached_memory_mb': mem_info['cache']
        }
    }

# --- Data Collection and Reporting Thread for Data Agents ---
def data_collection_thread():
    print(f"[{AGENT_ID}] Data collection thread started. Reporting to: {CENTRAL_MONITOR_URL}/metrics")
    while True:
        metrics_data = collect_metrics()
        try:
            # Send data to central monitor
            response = requests.post(f"{CENTRAL_MONITOR_URL}/metrics", json=metrics_data)
            response.raise_for_status() # Raise an exception for HTTP errors
            # print(f"[{AGENT_ID}] Metrics sent successfully.")
        except requests.exceptions.ConnectionError as e:
            print(f"[{AGENT_ID}] Failed to connect to central monitor at {CENTRAL_MONITOR_URL}: {e}")
        except requests.exceptions.RequestException as e:
            print(f"[{AGENT_ID}] Error sending metrics: {e}")
        except Exception as e:
            print(f"[{AGENT_ID}] An unexpected error occurred in data collection: {e}")

        time.sleep(5) # Collect and send data every 5 seconds

# --- Routes for Flask App ---

@app.route('/')
def index():
    # Only the central monitor will render the dashboard
    if RUN_DATA_COLLECTION:
        return "This is a Data Agent. It collects metrics and sends them to the central monitor."
    else:
        # Prepare data for rendering
        chart_data = {
            'cpu_user': [], 'cpu_system': [], 'free_memory': [], 'faults': [],
            'swap_in': [], 'swap_out': [], 'timestamps': []
        }
        anomalies_display = []
        recommendations_display = all_performance_recommendations

        # Aggregate data from all agents for charting
        # Get up to the last 100 points for each metric type across all agents
        # (You might want to refine this to show per-agent or aggregate more meaningfully for dashboard)
        for agent_id, history in all_agent_metrics.items():
            for entry in history[-100:]: # Limit data for display
                chart_data['timestamps'].append(f"{agent_id}@{datetime.fromisoformat(entry['timestamp']).strftime('%H:%M:%S')}")
                chart_data['cpu_user'].append(entry['metrics']['cpu_user'])
                chart_data['cpu_system'].append(entry['metrics']['cpu_system'])
                chart_data['free_memory'].append(entry['metrics']['free'])
                chart_data['faults'].append(entry['metrics']['faults'])
                chart_data['swap_in'].append(entry['metrics']['swap_in'])
                chart_data['swap_out'].append(entry['metrics']['swap_out'])

        # Prepare anomalies for display, including agent ID
        for anomaly in all_detected_anomalies:
            anomalies_display.append({
                'agent_id': anomaly.get('agent_id', 'N/A'),
                'timestamp': anomaly['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'score': f"{anomaly['score']:.2f}",
                'metrics': anomaly['metrics']
            })

        return render_template('index.html',
                               agent_id="Central Monitor",
                               chart_data=chart_data,
                               anomalies=anomalies_display,
                               recommendations=recommendations_display)

@app.route('/metrics', methods=['POST'])
def receive_metrics():
    # Only the central monitor processes incoming metrics
    if RUN_DATA_COLLECTION:
        return jsonify({"message": "This is a data agent, it does not receive metrics."}), 400
    
    data = request.get_json()
    if not data:
        return jsonify({"message": "No JSON data received"}), 400

    agent_id = data.get('agent_id', 'unknown')
    timestamp_str = data.get('timestamp')
    metrics = data.get('metrics')

    if not all([timestamp_str, metrics]):
        return jsonify({"message": "Missing timestamp or metrics data"}), 400

    timestamp = datetime.fromisoformat(timestamp_str)
    
    # Store metrics for this agent
    all_agent_metrics[agent_id].append({'timestamp': timestamp, 'metrics': metrics})

    # Keep history limited to prevent memory overflow
    if len(all_agent_metrics[agent_id]) > 1000:
        all_agent_metrics[agent_id] = all_agent_metrics[agent_id][-1000:]
    
    # Use the AI Analyzer instance for this specific agent
    current_analyzer = agent_analyzers[agent_id]
    current_analyzer.add_metrics(metrics)
    
    # Run anomaly detection and recommendations based on the agent's data
    detected_anomalies = current_analyzer.detect_anomalies()
    performance_recommendations = current_analyzer.get_performance_recommendations(metrics)

    # Append new anomalies with agent ID
    for anomaly in detected_anomalies:
        anomaly['agent_id'] = agent_id
        all_detected_anomalies.append(anomaly)
    # Keep global anomaly list limited
    if len(all_detected_anomalies) > 500:
        all_detected_anomalies = all_detected_anomalies[-500:]

    # Add new recommendations with agent ID
    # Avoid duplicates if checking frequently
    for rec in performance_recommendations:
        rec['agent_id'] = agent_id
        if rec not in all_performance_recommendations:
            all_performance_recommendations.append(rec)
    # Keep global recommendations list limited (you might want a more sophisticated dedupe/expiry)
    if len(all_performance_recommendations) > 100:
        all_performance_recommendations = all_performance_recommendations[-100:]

    return jsonify({"message": "Metrics received and processed"}), 200

@app.route('/healthz', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

# --- Start data collection thread if configured as an agent ---
if __name__ == '__main__':
    if RUN_DATA_COLLECTION:
        print(f"Starting application as Data Agent: {AGENT_ID}")
        data_thread = threading.Thread(target=data_collection_thread)
        data_thread.daemon = True # Allow main program to exit even if thread is running
        data_thread.start()
    else:
        print(f"Starting application as Central Monitor: {AGENT_ID}")
    app.run(host='0.0.0.0', port=5000, debug=False)