System Monitor with AI-Powered Anomaly Detection

Welcome! This project is a distributed system monitoring solution that gives you a real-time dashboard and AI-powered anomaly detection. It’s made up of a central monitor and multiple data agents, all running as Python Flask apps. The system collects key system metrics (like CPU, memory, swap, faults, and more) from multiple agents, brings them together in one place, and uses an AI model to spot anomalies and offer performance tips.

What can it do?

- Collects system metrics from multiple agents across your infrastructure.
- Shows you a real-time web dashboard with metrics, detected anomalies, and recommendations.
- Uses AI (a Keras autoencoder) to spot unusual system behavior based on what’s normal for your environment.
- Gives you actionable suggestions for optimizing CPU, memory, and swap usage.
- Ready for Kubernetes: comes with manifests for deploying everything in a cluster.
- Easy to run in Docker containers, too.

Metrics you’ll see:

- CPU usage (user, system, idle)
- Memory usage (free, cached, total)
- Swap usage (in/out, total, free)
- Page faults
- Network stats (bandwidth, packet loss, latency, error rate, TCP connections)

How does the AI help?

- It learns what “normal” looks like for your systems, then flags outliers using a sliding window autoencoder model.
- Calculates health scores for CPU, memory, network, storage, and application.
- Offers suggestions when things go out of bounds (like high CPU, low memory, or swap activity).

Quick Start (Local)

1. Install dependencies:
   pip install -r requirements.txt
2. Start the central monitor:
   (in your terminal)
   set RUN_DATA_COLLECTION=False
   set AGENT_ID=central-monitor
   set CENTRAL_MONITOR_URL=http://localhost:5000
   python app.py
3. Start a data agent (in a new terminal window):
   set RUN_DATA_COLLECTION=True
   set AGENT_ID=agent-1
   set CENTRAL_MONITOR_URL=http://localhost:5000
   python app.py
4. Open your browser and go to http://localhost:5000 (when running as central monitor).

Want to use Docker?

1. Build the Docker image:
   docker build -t system-monitor-app .
2. Run the central monitor:
   docker run -p 5000:5000 -e RUN_DATA_COLLECTION=False -e AGENT_ID=central-monitor -e CENTRAL_MONITOR_URL=http://localhost:5000 system-monitor-app
3. Run a data agent:
   docker run -e RUN_DATA_COLLECTION=True -e AGENT_ID=agent-1 -e CENTRAL_MONITOR_URL=http://host.docker.internal:5000 system-monitor-app

Kubernetes Deployment

- There’s a ConfigMap (monitor-config.yaml) that sets environment variables for agents and the monitor.
- The central monitor is defined in central-monitor.yaml.
- Data agents are defined in data-agents.yaml.

To deploy:
1. Create the ConfigMap:
   kubectl apply -f monitor-config.yaml
2. Deploy the central monitor:
   kubectl apply -f central-monitor.yaml
3. Deploy the data agents:
   kubectl apply -f data-agents.yaml
4. To access the dashboard, use kubectl get service central-monitor-service to find the NodePort and open it in your browser.

Note: The data agents use privileged mode in the example manifest to access system metrics. For production, you’ll want to look into more secure alternatives.

Configuration

- RUN_DATA_COLLECTION: True for agents, False for the central monitor.
- AGENT_ID: Give each agent a unique name.
- CENTRAL_MONITOR_URL: Where agents send their metrics.

What you need

- Python 3.8 or newer
- All Python dependencies are in requirements.txt
- Docker Engine if you want to use Docker
- kubectl and a running Kubernetes cluster (like Minikube) if you want to use Kubernetes

Project Structure

- app.py: Main Flask app for both agent and central monitor
- ai_analyzer.py: AI anomaly detection and recommendation logic
- templates/: HTML templates for the dashboard
- static/: CSS and static assets
- central-monitor.yaml, data-agents.yaml, monitor-config.yaml: Kubernetes manifests
- Dockerfile: Docker build instructions
- requirements.txt: Python dependencies

License

MIT License (add your details here)

---

This project is a foundation for scalable, AI-driven system monitoring. Contributions and suggestions are always welcome! 