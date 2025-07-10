from flask import Flask, jsonify, render_template, request, redirect, url_for
import subprocess
import pandas as pd
import time
from threading import Thread
import os
import psutil
import json
from datetime import datetime
import random
from collections import deque
import numpy as np
from sklearn.ensemble import IsolationForest
import threading
import requests

# Assuming ai_analyzer.py is in the same directory
from ai_analyzer import AIAnalyzer


app = Flask(__name__)

# --- Global Data Storage (Centralized for Aggregation) ---
# When running as a central monitor, this will store data from multiple agents.
# When running as an agent, this will store its local data before sending.
# Key structure: { 'agent_id': { 'metric_name': deque(maxlen=100), ... } }
# Or, if you want a flat structure, you'll need to add 'agent_id' to each dict in the deque.
# For simplicity, let's aggregate into a single deque for now, and you can add agent IDs later.
# For multi-agent setup, we'll store data keyed by agent_id
aggregated_metrics_history = {}
aggregated_network_history = {}

# AI analysis data storage (primarily on central monitor)
ai_history = {
    'timestamps': deque(maxlen=100),
    'anomaly_scores': deque(maxlen=100),
    'anomaly_score': 0,
    'previous_anomaly_score': 0,
    'prediction_accuracy': 0,
    'previous_accuracy': 0,
    'system_health': 0,
    'previous_health': 0,
    'active_alerts': 0,
    'previous_alerts': 0,
    'predicted_cpu': deque(maxlen=100),
    'actual_cpu': deque(maxlen=100),
    'cpu_health': 0,
    'memory_health': 0,
    'network_health': 0,
    'storage_health': 0,
    'application_health': 0,
    'anomalies': deque(maxlen=10),
    'recommendations': []
}


# --- Configuration Flags ---
RUN_DATA_COLLECTION = os.environ.get('RUN_DATA_COLLECTION', 'True') == 'True'
AGENT_ID = os.environ.get('AGENT_ID', 'local_agent')
CENTRAL_MONITOR_URL = os.environ.get('CENTRAL_MONITOR_URL', 'http://localhost:5000')

# Initialize AI Analyzer (will be used by central monitor to analyze aggregated data)
# Each agent can also have its own AI analyzer if needed for local anomaly detection.
ai_analyzer = AIAnalyzer()

# Global lists for vmstat and sar data for the *current* agent (if RUN_DATA_COLLECTION is True)
_local_vmstat_data = []
_local_vmstat_columns = None
_local_sar_fault_data = []


def _update_metrics_history(current_time, values, columns, fault_data_val):
    """Helper to update a single agent's local metrics history."""
    metrics_history = {
        'time': deque(maxlen=100),
        'cpu_user': deque(maxlen=100),
        'cpu_system': deque(maxlen=100),
        'free': deque(maxlen=100),
        'cache': deque(maxlen=100),
        'swap_in': deque(maxlen=100),
        'swap_out': deque(maxlen=100),
        'faults': deque(maxlen=100)
    }

    metrics_history['time'].append(current_time)
    metrics_history['cpu_user'].append(float(values[columns.index('us')]))
    metrics_history['cpu_system'].append(float(values[columns.index('sy')]))
    metrics_history['free'].append(float(values[columns.index('free')]) / (1024 * 1024))  # Convert to MB
    metrics_history['cache'].append(float(values[columns.index('cache')]) / (1024 * 1024))  # Convert to MB
    metrics_history['swap_in'].append(float(values[columns.index('si')]))
    metrics_history['swap_out'].append(float(values[columns.index('so')]))
    metrics_history['faults'].append(fault_data_val)

    return metrics_history


def collect_vmstat_data():
    """Collect data from vmstat every second."""
    global _local_vmstat_data, _local_vmstat_columns, _local_sar_fault_data
    process = subprocess.Popen(["vmstat", "1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while True:
        try:
            line = process.stdout.readline().decode("utf-8").strip()
            if line and not line.startswith("procs") and not line.startswith("r  b"):
                values = line.split()
                if not _local_vmstat_columns:
                    _local_vmstat_columns = [
                        "r", "b", "swpd", "free", "buff", "cache",
                        "si", "so", "bi", "bo", "in", "cs",
                        "us", "sy", "id", "wa", "st", "gu"
                    ]
                if len(values) == len(_local_vmstat_columns):
                    _local_vmstat_data.append(values)
                    if len(_local_vmstat_data) > 50:
                        _local_vmstat_data = _local_vmstat_data[-50:]

            time.sleep(1)
        except Exception as e:
            print(f"Error in collect_vmstat_data for agent {AGENT_ID}: {e}")
            time.sleep(1)

def collect_sar_faults():
    """Collect fault/s data from sar."""
    global _local_sar_fault_data
    process = subprocess.Popen(["sar", "-B", "1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    fault_index = None
    while True:
        line = process.stdout.readline().decode("utf-8").strip()
        if not line:
            continue
        if "Linux" in line or "Average" in line:
            continue
        if "fault/s" in line:
            headers = line.split()
            try:
                fault_index = headers.index("fault/s")
            except ValueError:
                fault_index = None
            continue
        if fault_index is None:
            continue
        tokens = line.split()
        if len(tokens) <= fault_index:
            continue
        try:
            fault_val = float(tokens[fault_index])
        except ValueError:
            continue
        _local_sar_fault_data.append(fault_val)
        if len(_local_sar_fault_data) > 50:
            _local_sar_fault_data = _local_sar_fault_data[-50:]
        time.sleep(1) # Added sleep to prevent busy-waiting if sar outputs slowly

def get_degree_of_multiprogramming():
    """Fetch Degree of Multiprogramming from /proc/loadavg."""
    try:
        with open('/proc/loadavg', 'r') as f:
            load_avg = f.read().split()
            degree = float(load_avg[0])
            return degree
    except Exception as e:
        print(f"Error reading /proc/loadavg: {e}")
        return 0

def parse_loadavg():
    """
    Read /proc/loadavg and return:
      - la1, la5, la15: 1-, 5-, 15-min load averages (floats)
      - running, total: number of runnable / total processes (ints)
      - last_pid: PID of most recently created process (int)
    """
    with open('/proc/loadavg') as f:
        fields = f.read().split()
    la1, la5, la15 = map(float, fields[:3])
    running, total = map(int, fields[3].split('/'))
    last_pid = int(fields[4])
    return la1, la5, la15, running, total, last_pid

# --- Network Data Collection (Agent-Side) ---
def collect_network_data_agent():
    """Collect network data and send to central monitor."""
    global _local_network_history # Reference to the local network history deque
    _local_network_history = {
        'timestamps': deque(maxlen=100),
        'bandwidth_usage': deque(maxlen=100),
        'packet_loss': deque(maxlen=100),
        'latency': deque(maxlen=100),
        'tcp_connections': deque(maxlen=100),
        'error_rate': deque(maxlen=100),
        'request_counts': {
            'GET': deque(maxlen=100),
            'POST': deque(maxlen=100),
            'PUT': deque(maxlen=100),
            'DELETE': deque(maxlen=100)
        },
        'interface_stats': {
            'rx_bytes': deque(maxlen=100),
            'tx_bytes': deque(maxlen=100),
            'rx_packets': deque(maxlen=100),
            'tx_packets': deque(maxlen=100),
            'rx_errors': deque(maxlen=100),
            'tx_errors': deque(maxlen=100)
        }
    }

    while True:
        try:
            now = time.time()
            _local_network_history['timestamps'].append(now)

            # Get network statistics from /proc/net/dev
            with open('/proc/net/dev', 'r') as f:
                lines = f.readlines()[2:]  # Skip header lines

            total_rx_bytes = 0
            total_tx_bytes = 0
            total_rx_packets = 0
            total_tx_packets = 0
            total_rx_errors = 0
            total_tx_errors = 0

            for line in lines:
                try:
                    iface = line.split(':')[0].strip()
                    if iface not in ['lo', 'docker0']:  # Skip loopback and docker interfaces
                        stats = line.split(':')[1].split()
                        if len(stats) >= 16:
                            total_rx_bytes += int(stats[0])
                            total_tx_bytes += int(stats[8])
                            total_rx_packets += int(stats[1])
                            total_tx_packets += int(stats[9])
                            total_rx_errors += int(stats[2])
                            total_tx_errors += int(stats[10])
                except (ValueError, IndexError):
                    continue

            # Update interface statistics
            _local_network_history['interface_stats']['rx_bytes'].append(total_rx_bytes)
            _local_network_history['interface_stats']['tx_bytes'].append(total_tx_bytes)
            _local_network_history['interface_stats']['rx_packets'].append(total_rx_packets)
            _local_network_history['interface_stats']['tx_packets'].append(total_tx_packets)
            _local_network_history['interface_stats']['rx_errors'].append(total_rx_errors)
            _local_network_history['interface_stats']['tx_errors'].append(total_tx_errors)

            # Calculate bandwidth usage in MB/s
            bandwidth = 0
            if len(_local_network_history['interface_stats']['rx_bytes']) > 1:
                rx_diff = total_rx_bytes - _local_network_history['interface_stats']['rx_bytes'][-2]
                tx_diff = total_tx_bytes - _local_network_history['interface_stats']['tx_bytes'][-2]
                bandwidth = (rx_diff + tx_diff) / (1024 * 1024)  # Convert to MB/s
            _local_network_history['bandwidth_usage'].append(bandwidth)

            # Calculate packet loss rate
            packet_loss = 0
            if total_rx_packets > 0:
                packet_loss = (total_rx_errors + total_tx_errors) / total_rx_packets * 100
            _local_network_history['packet_loss'].append(packet_loss)

            # Get TCP connection count
            try:
                with open('/proc/net/tcp', 'r') as f:
                    tcp_lines = f.readlines()[1:]  # Skip header
                _local_network_history['tcp_connections'].append(len(tcp_lines))
            except (FileNotFoundError, IOError):
                _local_network_history['tcp_connections'].append(0)

            # Calculate error rate
            error_rate = 0
            if total_rx_packets > 0:
                error_rate = (total_rx_errors + total_tx_errors) / total_rx_packets * 100
            _local_network_history['error_rate'].append(error_rate)

            # Measure actual network latency (agents should ping the central monitor or a known stable endpoint)
            latency = 0
            try:
                # Agents should ping the central monitor, or a known stable endpoint
                start_time = time.time()
                response = requests.get(f'{CENTRAL_MONITOR_URL}/ping_test', timeout=1)
                latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            except requests.exceptions.RequestException as e:
                print(f"Agent {AGENT_ID} - Error measuring latency to central monitor: {e}")
                latency = -1 # Indicate a failure to measure
            _local_network_history['latency'].append(latency)

            # Initialize request counts if empty (these are requests handled by this agent, not the central monitor)
            for method in ['GET', 'POST', 'PUT', 'DELETE']:
                if not _local_network_history['request_counts'][method]:
                    _local_network_history['request_counts'][method].append(0)

            # Send data to central monitor
            send_agent_data_to_central_monitor()

            time.sleep(5)  # Update every 5 seconds
        except Exception as e:
            print(f"Error collecting network data for agent {AGENT_ID}: {e}")
            time.sleep(5)

# --- Central Monitor Data Aggregation ---
def aggregate_agent_data(agent_id, metric_type, data_payload):
    """Aggregates data from a specific agent into the central monitor's storage."""
    global aggregated_metrics_history, aggregated_network_history

    if metric_type == 'system':
        if agent_id not in aggregated_metrics_history:
            aggregated_metrics_history[agent_id] = {
                'time': deque(maxlen=100),
                'cpu_user': deque(maxlen=100),
                'cpu_system': deque(maxlen=100),
                'free': deque(maxlen=100),
                'cache': deque(maxlen=100),
                'swap_in': deque(maxlen=100),
                'swap_out': deque(maxlen=100),
                'faults': deque(maxlen=100)
            }
        for key, value in data_payload.items():
            if key in aggregated_metrics_history[agent_id]:
                aggregated_metrics_history[agent_id][key].append(value)

    elif metric_type == 'network':
        if agent_id not in aggregated_network_history:
            aggregated_network_history[agent_id] = {
                'timestamps': deque(maxlen=100),
                'bandwidth_usage': deque(maxlen=100),
                'packet_loss': deque(maxlen=100),
                'latency': deque(maxlen=100),
                'tcp_connections': deque(maxlen=100),
                'error_rate': deque(maxlen=100),
                'request_counts': {
                    'GET': deque(maxlen=100),
                    'POST': deque(maxlen=100),
                    'PUT': deque(maxlen=100),
                    'DELETE': deque(maxlen=100)
                },
                'interface_stats': {
                    'rx_bytes': deque(maxlen=100),
                    'tx_bytes': deque(maxlen=100),
                    'rx_packets': deque(maxlen=100),
                    'tx_packets': deque(maxlen=100),
                    'rx_errors': deque(maxlen=100),
                    'tx_errors': deque(maxlen=100)
                }
            }
        for key, value in data_payload.items():
            if key in aggregated_network_history[agent_id]:
                if isinstance(value, dict): # For request_counts and interface_stats
                    for sub_key, sub_value in value.items():
                        if sub_key in aggregated_network_history[agent_id][key]:
                            aggregated_network_history[agent_id][key][sub_key].append(sub_value)
                else:
                    aggregated_network_history[agent_id][key].append(value)


def send_agent_data_to_central_monitor():
    """Sends collected local data from the agent to the central monitor."""
    global _local_vmstat_data, _local_vmstat_columns, _local_sar_fault_data, _local_network_history

    current_time = time.time()
    
    # Prepare system metrics
    system_metrics = {}
    if _local_vmstat_data and _local_vmstat_columns:
        last_vmstat = _local_vmstat_data[-1]
        system_metrics = {
            'time': current_time,
            'cpu_user': float(last_vmstat[_local_vmstat_columns.index('us')]),
            'cpu_system': float(last_vmstat[_local_vmstat_columns.index('sy')]),
            'free': float(last_vmstat[_local_vmstat_columns.index('free')]) / (1024 * 1024),
            'cache': float(last_vmstat[_local_vmstat_columns.index('cache')]) / (1024 * 1024),
            'swap_in': float(last_vmstat[_local_vmstat_columns.index('si')]),
            'swap_out': float(last_vmstat[_local_vmstat_columns.index('so')]),
            'faults': _local_sar_fault_data[-1] if _local_sar_fault_data else 0
        }
    
    # Prepare network metrics (ensure deques are converted to lists for JSON serialization)
    network_metrics = {key: list(value) if isinstance(value, deque) else {sub_key: list(sub_value) for sub_key, sub_value in value.items()} if isinstance(value, dict) else value for key, value in _local_network_history.items()}

    try:
        # Send system data
        if system_metrics:
            response = requests.post(
                f'{CENTRAL_MONITOR_URL}/receive_data',
                json={'agent_id': AGENT_ID, 'type': 'system', 'data': system_metrics},
                timeout=2
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            # print(f"Agent {AGENT_ID} sent system data. Status: {response.status_code}")

        # Send network data
        if network_metrics:
            response = requests.post(
                f'{CENTRAL_MONITOR_URL}/receive_data',
                json={'agent_id': AGENT_ID, 'type': 'network', 'data': network_metrics},
                timeout=2
            )
            response.raise_for_status()
            # print(f"Agent {AGENT_ID} sent network data. Status: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Agent {AGENT_ID} failed to send data to central monitor: {e}")


# --- AI Analysis on Central Monitor ---
def collect_ai_analysis_central():
    """Performs AI analysis on aggregated data on the central monitor."""
    global ai_history, aggregated_metrics_history, aggregated_network_history

    while True:
        try:
            now = time.time()
            ai_history['timestamps'].append(now)

            # Reset health indicators for recalculation based on *all* agents
            total_cpu_health = 0
            total_memory_health = 0
            total_network_health = 0
            total_agents = len(aggregated_metrics_history)
            
            anomalies_detected = []
            all_metrics_for_ai = []

            for agent_id, metrics_deque_dict in aggregated_metrics_history.items():
                if metrics_deque_dict['cpu_user']: # Ensure there's data
                    current_metrics = {
                        'cpu_user': metrics_deque_dict['cpu_user'][-1],
                        'cpu_system': metrics_deque_dict['cpu_system'][-1],
                        'free': metrics_deque_dict['free'][-1],
                        'cache': metrics_deque_dict['cache'][-1],
                        'swap_in': metrics_deque_dict['swap_in'][-1],
                        'swap_out': metrics_deque_dict['swap_out'][-1],
                        'faults': metrics_deque_dict['faults'][-1]
                    }
                    ai_analyzer.add_metrics(current_metrics)
                    anomalies_for_agent = ai_analyzer.detect_anomalies()
                    for anomaly in anomalies_for_agent:
                        anomalies_detected.append({**anomaly, 'agent_id': agent_id})
                    
                    # Update health indicators
                    cpu_usage = current_metrics['cpu_user'] + current_metrics['cpu_system']
                    # Assuming a total memory for accurate memory usage calculation
                    # For simplicity, let's assume `psutil.virtual_memory().total` on the central machine
                    # or better, agents should send their total memory.
                    # For now, a placeholder.
                    # memory_usage = 100 - (current_metrics['free'] / psutil.virtual_memory().total * 100) if psutil.virtual_memory().total > 0 else 0
                    memory_usage = random.uniform(20, 80) # Placeholder

                    total_cpu_health += max(0, 100 - cpu_usage)
                    total_memory_health += max(0, 100 - memory_usage)

                    # Add current metrics to a list for overall anomaly detection if needed
                    all_metrics_for_ai.append([
                        current_metrics['cpu_user'], current_metrics['cpu_system'],
                        current_metrics['free'], current_metrics['cache'],
                        current_metrics['swap_in'], current_metrics['swap_out'],
                        current_metrics['faults']
                    ])

            # Calculate average health across all agents
            if total_agents > 0:
                ai_history['cpu_health'] = total_cpu_health / total_agents
                ai_history['memory_health'] = total_memory_health / total_agents
            else:
                ai_history['cpu_health'] = 100
                ai_history['memory_health'] = 100

            # Network health can be derived from aggregated_network_history
            total_network_latency = 0
            total_network_packet_loss = 0
            total_network_error_rate = 0
            network_agents_count = len(aggregated_network_history)

            if network_agents_count > 0:
                for agent_id, network_deque_dict in aggregated_network_history.items():
                    if network_deque_dict['latency']:
                        total_network_latency += network_deque_dict['latency'][-1]
                    if network_deque_dict['packet_loss']:
                        total_network_packet_loss += network_deque_dict['packet_loss'][-1]
                    if network_deque_dict['error_rate']:
                        total_network_error_rate += network_deque_dict['error_rate'][-1]
                
                avg_latency = total_network_latency / network_agents_count
                avg_packet_loss = total_network_packet_loss / network_agents_count
                avg_error_rate = total_network_error_rate / network_agents_count

                # Simple scoring for network health (adjust thresholds as needed)
                network_health_score = 100
                if avg_latency > 100: # High latency
                    network_health_score -= (avg_latency / 100) * 10
                if avg_packet_loss > 1: # Some packet loss
                    network_health_score -= avg_packet_loss * 5
                if avg_error_rate > 0.5: # Some errors
                    network_health_score -= avg_error_rate * 10

                ai_history['network_health'] = max(0, min(100, network_health_score))
            else:
                ai_history['network_health'] = 100 # No network agents, assume healthy

            ai_history['storage_health'] = max(0, min(100, random.uniform(80, 100))) # Placeholder
            ai_history['application_health'] = max(0, min(100, random.uniform(80, 100))) # Placeholder

            # Update overall system health
            ai_history['system_health'] = (
                ai_history['cpu_health'] +
                ai_history['memory_health'] +
                ai_history['network_health'] +
                ai_history['storage_health'] +
                ai_history['application_health']
            ) / 5

            # Update anomaly score based on system health
            ai_history['previous_anomaly_score'] = ai_history['anomaly_score']
            ai_history['anomaly_score'] = 1 - (ai_history['system_health'] / 100)
            ai_history['anomaly_scores'].append(ai_history['anomaly_score'])

            # Update prediction accuracy (simulated for now)
            ai_history['previous_accuracy'] = ai_history['prediction_accuracy']
            ai_history['prediction_accuracy'] = random.uniform(85, 99)

            # Update active alerts based on detected anomalies
            ai_history['anomalies'].clear()
            for anomaly in anomalies_detected:
                ai_history['anomalies'].append(anomaly)
            ai_history['active_alerts'] = len(ai_history['anomalies'])

            # Update CPU predictions (simulated, ideally use a predictive model)
            if all_metrics_for_ai and len(all_metrics_for_ai) > 0:
                avg_cpu_user = np.mean([m[0] for m in all_metrics_for_ai]) # Assuming cpu_user is first
                ai_history['predicted_cpu'].append(max(0, min(100, avg_cpu_user + random.uniform(-5, 5))))
                ai_history['actual_cpu'].append(avg_cpu_user)
            else:
                ai_history['predicted_cpu'].append(0)
                ai_history['actual_cpu'].append(0)


            # Update recommendations (based on aggregated data and health)
            ai_history['recommendations'] = ai_analyzer.get_performance_recommendations({
                'cpu_user': ai_history['actual_cpu'][-1] if ai_history['actual_cpu'] else 0,
                'cpu_system': ai_history['actual_cpu'][-1] if ai_history['actual_cpu'] else 0, # Placeholder
                'free': ai_history['memory_health'] if ai_history['memory_health'] else 0, # Invert health for 'free' metric
                'cache': 0, # Placeholder
                'swap_in': 0, # Placeholder
                'swap_out': 0, # Placeholder
                'faults': 0 # Placeholder
            })

            time.sleep(5) # Analyze every 5 seconds
        except Exception as e:
            print(f"Error in central AI analysis: {e}")
            time.sleep(5)


# --- Flask Routes ---
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """System performance dashboard"""
    return render_template('ebpf_dashboard.html')

@app.route('/network')
def network_dashboard():
    """Network performance dashboard"""
    return render_template('network.html')

@app.route('/thrash')
def thrash_dashboard():
    """Memory thrashing dashboard"""
    return render_template('thrash.html')

@app.route('/ai')
def ai_dashboard():
    return render_template('ai.html')

@app.route('/data')
def get_data():
    """Get system metrics data (aggregated on central monitor or local on agent)"""
    # If this is the central monitor, return aggregated data
    if not RUN_DATA_COLLECTION:
        # Flatten the aggregated data for easier charting on the UI
        # This assumes the UI expects single series for each metric
        # You might need to adjust your JS to handle per-agent data
        flattened_data = {
            "time": [],
            "free": [],
            "cache": [],
            "cpu_user": [],
            "cpu_system": [],
            "cpu_idle": [],
            "swap_in": [],
            "swap_out": [],
            "faults": [],
            "agents": list(aggregated_metrics_history.keys())
        }
        
        # This is a simplified flattening. For actual multi-agent visualization,
        # you'd likely want to send data per agent or calculate averages/sums.
        # Here, we'll just take the latest from a random agent for demo.
        if aggregated_metrics_history:
            # You'd iterate through aggregated_metrics_history and extract data for each agent
            # For a simple demo, let's just pick one or average them.
            # Example: picking the first agent found
            first_agent_id = list(aggregated_metrics_history.keys())[0]
            agent_data = aggregated_metrics_history[first_agent_id]
            
            # Since 'time' is typically shared, we can use one agent's time
            flattened_data["time"] = list(range(len(agent_data["free"]))) # Or use actual timestamps if consistent

            for key in ["free", "cache", "cpu_user", "cpu_system", "swap_in", "swap_out", "faults"]:
                if key in agent_data:
                    flattened_data[key] = list(agent_data[key])
                else:
                    flattened_data[key] = [0] * len(flattened_data["time"]) # Fill with zeros if not present
            
            # cpu_idle is not directly collected by vmstat, derive or set to 0 for now.
            flattened_data["cpu_idle"] = [0] * len(flattened_data["time"]) # Placeholder for now

        return jsonify(flattened_data)
    else:
        # If this is an agent, return its local data (for testing agent UI directly)
        if not _local_vmstat_data or not _local_vmstat_columns:
            return jsonify({
                "time": [],
                "free": [],
                "cache": [],
                "cpu_user": [],
                "cpu_system": [],
                "cpu_idle": [],
                "swap_in": [],
                "swap_out": [],
                "faults": []
            })
        
        df = pd.DataFrame(_local_vmstat_data, columns=_local_vmstat_columns).apply(pd.to_numeric, errors="ignore")
        
        response_data = {
            "time": list(range(len(_local_vmstat_data))),
            "free": list(df["free"]),
            "cache": list(df["cache"]),
            "cpu_user": list(df["us"]),
            "cpu_system": list(df["sy"]),
            "cpu_idle": list(df["id"]),
            "swap_in": list(df["si"]),
            "swap_out": list(df["so"]),
            "faults": _local_sar_fault_data
        }
        return jsonify(response_data)


@app.route('/thrash-data')
def thrash_data():
    """Get memory thrashing data (local on agent or average on central)"""
    if not RUN_DATA_COLLECTION:
        # Central monitor logic: return aggregated/averaged thrash data
        # For simplicity, let's average the fault data from all agents
        all_faults = []
        for agent_id, metrics in aggregated_metrics_history.items():
            all_faults.extend(list(metrics['faults'])) # Collect all fault data

        avg_faults = []
        if all_faults:
            # This is a very simplistic average. You might want to align timestamps.
            # For now, let's just use the last 'n' values across all agents or calculate a rolling average.
            # Example: average the last value from each agent
            latest_faults = [metrics['faults'][-1] for agent_id, metrics in aggregated_metrics_history.items() if metrics['faults']]
            if latest_faults:
                avg_faults.append(np.mean(latest_faults))
            else:
                avg_faults.append(0)
            
            # For thrash-data route, typically it expects a list. If you want a historical trend,
            # you need a more sophisticated aggregation strategy for 'faults' specifically.
            # For now, let's return the latest available values for simplification.
            # You might need to redesign your frontend to handle historical multi-agent data.
            avg_faults = list(np.mean([list(metrics['faults']) for agent_id, metrics in aggregated_metrics_history.items() if metrics['faults']], axis=0)) if aggregated_metrics_history and any(metrics['faults'] for metrics in aggregated_metrics_history.values()) else [0]
            
        la1, la5, la15, running, total, last_pid = parse_loadavg() # This will be the central monitor's loadavg
        cpu_utilization = ai_history['actual_cpu'][-1] if ai_history['actual_cpu'] else 0 # Use overall CPU from AI analysis

        return jsonify({
            "time": list(range(len(avg_faults))), # Time needs to align with `avg_faults` length
            "faults": avg_faults,
            "degree": float(la1), # Using 1-min load average as degree
            "cpu_utilization": float(cpu_utilization),
            "loadavg1": la1,
            "loadavg5": la5,
            "loadavg15": la15,
            "running": running,
            "total": total,
            "last_pid": last_pid
        })
    else:
        # Agent's local thrash data
        la1, la5, la15, running, total, last_pid = parse_loadavg()
        # Calculate CPU utilization for the agent itself
        cpu_utilization = 0
        if _local_vmstat_data and _local_vmstat_columns:
            last_vmstat = _local_vmstat_data[-1]
            cpu_utilization = float(last_vmstat[_local_vmstat_columns.index('us')]) + float(last_vmstat[_local_vmstat_columns.index('sy')])

        return jsonify({
            "time": list(range(len(_local_sar_fault_data))),
            "faults": _local_sar_fault_data,
            "degree": float(get_degree_of_multiprogramming()),
            "cpu_utilization": float(cpu_utilization),
            "loadavg1": la1,
            "loadavg5": la5,
            "loadavg15": la15,
            "running": running,
            "total": total,
            "last_pid": last_pid
        })


@app.route('/ai-analysis')
def get_ai_analysis():
    """Get AI-based analysis and recommendations (from central monitor)"""
    return jsonify({
        'timestamps': list(ai_history['timestamps']),
        'anomaly_scores': list(ai_history['anomaly_scores']),
        'anomaly_score': ai_history['anomaly_score'],
        'previous_anomaly_score': ai_history['previous_anomaly_score'],
        'prediction_accuracy': ai_history['prediction_accuracy'],
        'previous_accuracy': ai_history['previous_accuracy'],
        'system_health': ai_history['system_health'],
        'previous_health': ai_history['previous_health'],
        'active_alerts': ai_history['active_alerts'],
        'previous_alerts': ai_history['previous_alerts'],
        'predicted_cpu': list(ai_history['predicted_cpu']),
        'actual_cpu': list(ai_history['actual_cpu']),
        'cpu_health': ai_history['cpu_health'],
        'memory_health': ai_history['memory_health'],
        'network_health': ai_history['network_health'],
        'storage_health': ai_history['storage_health'],
        'application_health': ai_history['application_health'],
        'anomalies': list(ai_history['anomalies']),
        'recommendations': ai_history['recommendations'],
        'current_metrics': {} # This should be derived from overall aggregated data if needed for display
    })

@app.route('/network-data')
def get_network_data():
    """Get network metrics data (aggregated on central monitor or local on agent)"""
    if not RUN_DATA_COLLECTION:
        # Central monitor: return aggregated network data
        # For simplicity, returning a combined view or average.
        # You'll likely need to send per-agent data to the frontend for detailed views.
        flattened_network_data = {
            'timestamps': deque(maxlen=100),
            'bandwidth_usage': deque(maxlen=100),
            'packet_loss': deque(maxlen=100),
            'latency': deque(maxlen=100),
            'tcp_connections': deque(maxlen=100),
            'error_rate': deque(maxlen=100),
            'request_counts': {
                'GET': deque(maxlen=100),
                'POST': deque(maxlen=100),
                'PUT': deque(maxlen=100),
                'DELETE': deque(maxlen=100)
            },
            'interface_stats': {
                'rx_bytes': deque(maxlen=100),
                'tx_bytes': deque(maxlen=100),
                'rx_packets': deque(maxlen=100),
                'tx_packets': deque(maxlen=100),
                'rx_errors': deque(maxlen=100),
                'tx_errors': deque(maxlen=100)
            },
            'agents': list(aggregated_network_history.keys())
        }

        if aggregated_network_history:
            # Averages or sums across all agents
            # This is a very simplified aggregation. For real dashboards, you'd want per-agent graphs
            # or more complex calculations (e.g., total bandwidth, average latency).
            num_agents = len(aggregated_network_history)
            if num_agents > 0:
                # Take the last available timestamp from any agent
                first_agent_id = list(aggregated_network_history.keys())[0]
                flattened_network_data['timestamps'] = list(aggregated_network_history[first_agent_id]['timestamps'])

                for key in ['bandwidth_usage', 'packet_loss', 'latency', 'tcp_connections', 'error_rate']:
                    # Calculate average for numerical metrics across agents
                    avg_values = [np.mean([d[key][i] for d in aggregated_network_history.values() if len(d[key]) > i])
                                  for i in range(len(flattened_network_data['timestamps']))]
                    flattened_network_data[key] = list(avg_values)
                
                # For request counts and interface stats, sum across all agents
                for req_type in ['GET', 'POST', 'PUT', 'DELETE']:
                    sum_values = [sum([d['request_counts'][req_type][i] for d in aggregated_network_history.values() if len(d['request_counts'][req_type]) > i])
                                  for i in range(len(flattened_network_data['timestamps']))]
                    flattened_network_data['request_counts'][req_type] = list(sum_values)
                
                for stat_type in ['rx_bytes', 'tx_bytes', 'rx_packets', 'tx_packets', 'rx_errors', 'tx_errors']:
                    sum_values = [sum([d['interface_stats'][stat_type][i] for d in aggregated_network_history.values() if len(d['interface_stats'][stat_type]) > i])
                                  for i in range(len(flattened_network_data['timestamps']))]
                    flattened_network_data['interface_stats'][stat_type] = list(sum_values)

        return jsonify({key: list(value) if isinstance(value, deque) else {sk: list(sv) for sk, sv in value.items()} for key, value in flattened_network_data.items()})

    else:
        # Agent's local network data
        return jsonify({
            'timestamps': list(_local_network_history['timestamps']),
            'bandwidth_usage': list(_local_network_history['bandwidth_usage']),
            'packet_loss': list(_local_network_history['packet_loss']),
            'latency': list(_local_network_history['latency']),
            'tcp_connections': list(_local_network_history['tcp_connections']),
            'error_rate': list(_local_network_history['error_rate']),
            'request_counts': {
                'GET': list(_local_network_history['request_counts']['GET']),
                'POST': list(_local_network_history['request_counts']['POST']),
                'PUT': list(_local_network_history['request_counts']['PUT']),
                'DELETE': list(_local_network_history['request_counts']['DELETE'])
            },
            'interface_stats': {
                'rx_bytes': list(_local_network_history['interface_stats']['rx_bytes']),
                'tx_bytes': list(_local_network_history['interface_stats']['tx_bytes']),
                'rx_packets': list(_local_network_history['interface_stats']['rx_packets']),
                'tx_packets': list(_local_network_history['interface_stats']['tx_packets']),
                'rx_errors': list(_local_network_history['interface_stats']['rx_errors']),
                'tx_errors': list(_local_network_history['interface_stats']['tx_errors'])
            }
        })


@app.route('/receive_data', methods=['POST'])
def receive_data():
    """Endpoint for agents to send their data to the central monitor."""
    if RUN_DATA_COLLECTION: # This instance should not receive data if it's an agent
        return jsonify({"status": "error", "message": "This instance is a data agent, not a central monitor."}), 400

    data = request.get_json()
    agent_id = data.get('agent_id')
    metric_type = data.get('type')
    payload = data.get('data')

    if not agent_id or not metric_type or not payload:
        return jsonify({"status": "error", "message": "Missing agent_id, type, or data"}), 400

    aggregate_agent_data(agent_id, metric_type, payload)
    return jsonify({"status": "success", "message": f"Received {metric_type} data from {agent_id}"})


@app.route('/ping_test')
def ping_test():
    """A simple endpoint for agents to test latency to the central monitor."""
    return jsonify({"status": "pong"})


@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def catch_all_requests(path):
    """Catch-all for simulating requests to the agent, updating request counts."""
    if RUN_DATA_COLLECTION: # Only agents should track their own request counts
        method = request.method
        if method in _local_network_history['request_counts']:
            # Get the last count or 0 if empty
            last_count = _local_network_history['request_counts'][method][-1] if _local_network_history['request_counts'][method] else 0
            # Append the new count
            _local_network_history['request_counts'][method].append(last_count + 1)
        return jsonify({"status": "success", "method": method, "agent_id": AGENT_ID})
    else: # Central monitor should not track requests directly
        return jsonify({"status": "info", "message": "This is the central monitor, not tracking direct requests."})


if __name__ == '__main__':
    if RUN_DATA_COLLECTION:
        print(f"Starting as Data Agent: {AGENT_ID}")
        # Start local data collection threads for this agent
        vmstat_thread = Thread(target=collect_vmstat_data)
        vmstat_thread.daemon = True
        vmstat_thread.start()

        sar_thread = Thread(target=collect_sar_faults)
        sar_thread.daemon = True
        sar_thread.start()

        network_thread = Thread(target=collect_network_data_agent)
        network_thread.daemon = True
        network_thread.start()

        # Agents don't run AI analysis or receive data from others
    else:
        print("Starting as Central Monitor")
        # Central monitor runs AI analysis on aggregated data
        ai_analysis_thread = Thread(target=collect_ai_analysis_central)
        ai_analysis_thread.daemon = True
        ai_analysis_thread.start()

    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)