import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Import for Autoencoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf 

class AIAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.metric_history = []
        self.window_size = 10  # Number of samples to consider for anomaly detection

        # Autoencoder parameters
        self.autoencoder = None
        self.encoding_dim = 4 # Dimension of the latent space, smaller than input features
        self.autoencoder_epochs = 50 # Number of training epochs for the autoencoder
        self.autoencoder_batch_size = 16 # Batch size for training (can be adjusted)
        self.autoencoder_reconstruction_error_threshold = None # This will be learned
        self.autoencoder_initial_training_data_multiplier = 5 # Need window_size * this many samples for initial training
        self.autoencoder_trained = False # Flag to track if AE is initially trained

    def _build_autoencoder_model(self, input_dim):
        """Builds a simple Autoencoder model."""
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoder = Dense(self.encoding_dim * 2, activation='relu')(input_layer)
        encoder = Dense(self.encoding_dim, activation='relu')(encoder)
        
        # Decoder
        decoder = Dense(self.encoding_dim * 2, activation='relu')(encoder)
        decoder = Dense(input_dim, activation='linear')(decoder) # Output layer matches input dim
        
        autoencoder_model = Model(inputs=input_layer, outputs=decoder)
        autoencoder_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return autoencoder_model
        
    def _get_window_metrics_df(self, history_subset):
        """Helper to convert a history subset into a DataFrame of window-aggregated metrics."""
        window_data = []
        for i in range(len(history_subset) - self.window_size + 1):
            window = history_subset[i:i + self.window_size]
            window_metrics = {
                'cpu_user_mean': np.mean([m['metrics']['cpu_user'] for m in window]),
                'cpu_system_mean': np.mean([m['metrics']['cpu_system'] for m in window]),
                'free_memory_mean': np.mean([m['metrics']['free'] for m in window]),
                'faults_mean': np.mean([m['metrics']['faults'] for m in window]),
                'swap_in_mean': np.mean([m['metrics']['swap_in'] for m in window]),
                'swap_out_mean': np.mean([m['metrics']['swap_out'] for m in window])
            }
            window_data.append(window_metrics)
        return pd.DataFrame(window_data)

    def add_metrics(self, metrics):
        """Add new metrics to the history"""
        self.metric_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })
        # Keep history limited to prevent memory overflow
        if len(self.metric_history) > 1000:
            self.metric_history = self.metric_history[-1000:]
    
    def detect_anomalies(self):
        """Detect anomalies in the collected metrics using Autoencoder (sliding window approach)"""
        # Ensure we have enough data for at least one window
        if len(self.metric_history) < self.window_size:
            return []
            
        # --- Initial Autoencoder Training ---
        if not self.autoencoder_trained:
            required_initial_samples = self.window_size * self.autoencoder_initial_training_data_multiplier
            if len(self.metric_history) >= required_initial_samples:
                print(f"[{datetime.now().isoformat()}] Initializing Autoencoder training with first {required_initial_samples} samples...")
                
                # Get aggregated window data from the initial set of samples
                initial_training_df = self._get_window_metrics_df(self.metric_history[:required_initial_samples])
                
                if initial_training_df.empty:
                    print(f"[{datetime.now().isoformat()}] Not enough valid window data for initial AE training. Returning empty anomalies.")
                    return []

                # Fit scaler on this initial 'normal' training data
                scaled_initial_data = self.scaler.fit_transform(initial_training_df)

                # Build autoencoder model
                input_dim = scaled_initial_data.shape[1]
                self.autoencoder = self._build_autoencoder_model(input_dim)
                
                # Train the autoencoder
                self.autoencoder.fit(scaled_initial_data, scaled_initial_data, 
                                     epochs=self.autoencoder_epochs, 
                                     batch_size=self.autoencoder_batch_size, 
                                     shuffle=True, verbose=0)
                
                # Calculate reconstruction errors on the training data to set threshold
                reconstructions = self.autoencoder.predict(scaled_initial_data, verbose=0)
                initial_reconstruction_errors = np.mean(np.square(scaled_initial_data - reconstructions), axis=1)

                # Set the threshold as the 95th percentile of reconstruction errors from normal data
                # Adjust percentile if too many/few anomalies
                self.autoencoder_reconstruction_error_threshold = np.percentile(initial_reconstruction_errors, 95)
                self.autoencoder_trained = True
                print(f"[{datetime.now().isoformat()}] Autoencoder trained. Reconstruction error threshold set to: {self.autoencoder_reconstruction_error_threshold:.4f}")
            else:
                # Not enough data for initial training yet
                return [] 

        # --- Anomaly Detection using Trained Autoencoder ---
        if self.autoencoder_trained:
            # Get the latest window's aggregated metrics
            current_window_df = self._get_window_metrics_df(self.metric_history[-self.window_size:])
            
            if current_window_df.empty:
                print(f"[{datetime.now().isoformat()}] Current window data is empty. Returning empty anomalies.")
                return []

            # Transform current data using the *already fitted* scaler
            scaled_current_window_data = self.scaler.transform(current_window_df)
            
            # Predict reconstruction for the current window
            reconstructions = self.autoencoder.predict(scaled_current_window_data, verbose=0)
            
            # Calculate reconstruction error for the current window
            reconstruction_errors = np.mean(np.square(scaled_current_window_data - reconstructions), axis=1)
            
            anomalies = []
            # Check if the latest reconstruction error exceeds the learned threshold
            # Since we are processing one window (the latest), reconstruction_errors will have one value
            if reconstruction_errors[0] > self.autoencoder_reconstruction_error_threshold:
                # The anomaly is linked to the metrics of the *last* sample in the anomalous window
                anomalous_metric_entry = self.metric_history[-1] 
                anomalies.append({
                    'timestamp': anomalous_metric_entry['timestamp'],
                    'metrics': anomalous_metric_entry['metrics'],
                    'score': reconstruction_errors[0], # Using reconstruction error as the score
                    'anomaly_type': 'Autoencoder Reconstruction Error'
                })
            return anomalies
        
        return [] # Fallback if for some reason AE is not trained yet (shouldn't happen if logic is followed)

    def get_performance_recommendations(self, current_metrics):
        """Generate performance optimization recommendations based on system metrics"""
        recommendations = []
        
        # CPU recommendations
        if current_metrics.get('cpu_user', 0) > 80:
            recommendations.append({
                'type': 'cpu',
                'severity': 'high',
                'message': 'High CPU user time detected. Consider optimizing application code or scaling resources.',
                'metric': 'cpu_user',
                'value': current_metrics['cpu_user']
            })
        
        if current_metrics.get('cpu_system', 0) > 30:
            recommendations.append({
                'type': 'cpu',
                'severity': 'medium',
                'message': 'High system CPU usage detected. Check for system-level bottlenecks.',
                'metric': 'cpu_system',
                'value': current_metrics['cpu_system']
            })
        
        # Memory recommendations
        if current_metrics.get('free', 0) < 1000:  # Assuming MB
            recommendations.append({
                'type': 'memory',
                'severity': 'high',
                'message': 'Low free memory detected. Consider memory optimization or increasing resources.',
                'metric': 'free',
                'value': current_metrics['free']
            })
        
        if current_metrics.get('cached_memory_mb', 0) > 5000:  # Assuming MB
            recommendations.append({
                'type': 'memory',
                'severity': 'medium',
                'message': 'High cache usage detected. Consider tuning cache parameters.',
                'metric': 'cache',
                'value': current_metrics['cached_memory_mb']
            })
        
        # Swap recommendations
        if current_metrics.get('swap_in', 0) > 0 or current_metrics.get('swap_out', 0) > 0:
            recommendations.append({
                'type': 'memory',
                'severity': 'high',
                'message': 'Swap activity detected. System may be under memory pressure.',
                'metric': 'swap',
                'value': current_metrics['swap_in'] + current_metrics['swap_out']
            })
        
        # Fault recommendations
        if current_metrics.get('faults', 0) > 1000:  # Assuming faults per second
            recommendations.append({
                'type': 'memory',
                'severity': 'medium',
                'message': 'High page fault rate detected. Consider memory optimization.',
                'metric': 'faults',
                'value': current_metrics['faults']
            })
        
        return recommendations