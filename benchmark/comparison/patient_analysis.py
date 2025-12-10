"""
Patient-level analysis for understanding performance differences within experiments.

This module provides tools for:
- Analyzing which patients perform better/worse
- t-SNE visualization of patient characteristics
- Correlation analysis between patient features and model performance
- Statistical analysis of patient subgroups

UPDATED: This module now uses pre-computed metrics from comprehensive_metrics_*.json files
instead of calculating metrics from raw predictions. It focuses only on visualization and
analysis of existing results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional
import json


class PatientAnalyzer:
    """
    Analyze patient-level performance differences within a single experiment.
    """
    
    def __init__(self, experiment_dir: str, experiment_name: str = None):
        """
        Initialize the patient analyzer.
        
        Args:
            experiment_dir: Path to experiment directory
            experiment_name: Optional name for the experiment
        """
        print(experiment_dir)
        self.experiment_dir = Path(experiment_dir)
        self.experiment_name = experiment_name or self.experiment_dir.name
        
        self.patient_results = {}
        self.patient_features = {}
        self.performance_data = None
        
    def load_experiment_results(self):
        """Load results from the comprehensive metrics JSON file."""
        print(f"Loading patient results from {self.experiment_name}...")
        
        # Find comprehensive metrics JSON files
        json_files = list(self.experiment_dir.glob("comprehensive_metrics_*.json"))
        
        if not json_files:
            print(f"    No comprehensive_metrics_*.json files found in {self.experiment_dir}")
            return self.patient_results
        
        # Use the most recent JSON file if multiple exist
        json_file = sorted(json_files)[-1]
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Load from comprehensive_metrics format
            patient_metrics = data.get('patient_metrics', {})
            
            for patient_id_str, patient_data in patient_metrics.items():
                try:
                    patient_id = int(patient_id_str)
                except:
                    continue
                
                if patient_id not in self.patient_results:
                    self.patient_results[patient_id] = {}
                
                for model_name, model_data in patient_data.items():
                    # Extract all metrics directly from comprehensive JSON
                    mae = model_data.get('mae')
                    rmse = model_data.get('rmse')
                    mape = model_data.get('mape')
                    mard = model_data.get('mard')
                    
                    # Extract TIR metrics
                    tir_data = model_data.get('tir', {})
                    tir = tir_data.get('time_in_range', 0) if isinstance(tir_data, dict) else 0
                    hypo_events = tir_data.get('time_below_range', 0) if isinstance(tir_data, dict) else 0
                    hyper_events = tir_data.get('time_above_range', 0) if isinstance(tir_data, dict) else 0
                    
                    # Extract Clarke and Parkes zones
                    clarke_zones = model_data.get('clarke_zones', {})
                    clarke_a_b = model_data.get('clarke_a_b', 0)
                    parkes_zones = model_data.get('parkes_zones', {})
                    parkes_a_b = model_data.get('parkes_a_b', 0)
                    
                    # Load glucose data (y_true) from predictions CSV for patient feature computation
                    y_true = None
                    n_predictions = 0
                    patient_dir = self.experiment_dir / f"patient_{patient_id}"
                    if patient_dir.exists():
                        pred_files = list(patient_dir.glob(f"{model_name}_patient_{patient_id}_predictions.csv"))
                        if pred_files:
                            pred_file = pred_files[0]
                            try:
                                df = pd.read_csv(pred_file)
                                if 'true_values' in df.columns:
                                    y_true = df['true_values'].values
                                elif 'actual' in df.columns:
                                    y_true = df['actual'].values
                                else:
                                    # Try to infer column names - use first column as true values
                                    cols = df.columns.tolist()
                                    if len(cols) >= 1:
                                        y_true = df.iloc[:, 0].values
                                
                                if y_true is not None:
                                    n_predictions = len(y_true)
                                    
                            except Exception as e:
                                print(f"    Warning: Could not load glucose data from {pred_file}: {e}")
                    
                    self.patient_results[patient_id][model_name] = {
                        'mae': mae,
                        'rmse': rmse,
                        'mape': mape,
                        'mard': mard,
                        'tir': tir,
                        'hypo_events': hypo_events,
                        'hyper_events': hyper_events,
                        'clarke_a_b': clarke_a_b,
                        'parkes_a_b': parkes_a_b,
                        'clarke_zones': clarke_zones,
                        'parkes_zones': parkes_zones,
                        'y_true': y_true,  # Only for patient feature computation
                        'n_predictions': n_predictions
                    }
            
            print(f"[OK] Loaded {len(self.patient_results)} patients from {json_file.name}")
            
        except Exception as e:
            print(f"    Error loading JSON file {json_file}: {e}")
        
        return self.patient_results
    
    def compute_patient_features(self):
        """
        Compute comprehensive features for each patient based on their glucose data.
        Only extracts patient characteristics, not performance metrics.
        """
        print("Computing patient features...")
        
        for patient_id in self.patient_results.keys():
            # Get glucose data from first model available (should be the same for all models)
            first_model = list(self.patient_results[patient_id].keys())[0]
            y_true = self.patient_results[patient_id][first_model]['y_true']
            
            if y_true is None or len(y_true) == 0:
                print(f"    Warning: No glucose data available for patient {patient_id}")
                continue
            
            # Basic statistical features
            mean_glucose = np.mean(y_true)
            std_glucose = np.std(y_true)
            
            # Compute glucose dynamics features
            glucose_diff = np.diff(y_true)
            mean_change = np.mean(glucose_diff)
            std_change = np.std(glucose_diff)
            
            # Range and quartiles
            glucose_range = np.max(y_true) - np.min(y_true)
            iqr = np.percentile(y_true, 75) - np.percentile(y_true, 25)
                        
            # Time-based features
            n_samples = len(y_true)
            
            # Clinical ranges
            hypo_percent = np.sum(y_true < 70) / len(y_true) * 100
            hyper_percent = np.sum(y_true > 180) / len(y_true) * 100
            in_range_percent = np.sum((y_true >= 70) & (y_true <= 180)) / len(y_true) * 100
            severe_hypo_percent = np.sum(y_true < 54) / len(y_true) * 100
            severe_hyper_percent = np.sum(y_true > 250) / len(y_true) * 100
            
            # Glucose stability metrics
            n_rapid_changes = np.sum(np.abs(glucose_diff) > 20)  # Changes > 20 mg/dL
            stability_score = 100 - (n_rapid_changes / len(glucose_diff) * 100)
            
            features = {
                'mean_glucose': mean_glucose,
                'std_glucose': std_glucose,
                'min_glucose': np.min(y_true),
                'max_glucose': np.max(y_true),
                'median_glucose': np.median(y_true),
                'q25_glucose': np.percentile(y_true, 25),
                'q75_glucose': np.percentile(y_true, 75),
                'glucose_range': glucose_range,
                'iqr_glucose': iqr,
                'mean_change': mean_change,
                'std_change': std_change,
                'n_samples': n_samples,
                'hypo_percent': hypo_percent,
                'hyper_percent': hyper_percent,
                'in_range_percent': in_range_percent,
                'severe_hypo_percent': severe_hypo_percent,
                'severe_hyper_percent': severe_hyper_percent,
                'n_rapid_changes': n_rapid_changes,
                'stability_score': stability_score
            }
            
            self.patient_features[patient_id] = features
        
        print(f"[OK] Computed features for {len(self.patient_features)} patients")
    
    def create_performance_dataframe(self, model_name: str = None):
        """
        Create a comprehensive DataFrame with patient features and performance metrics.
        
        Args:
            model_name: Specific model to analyze. If None, uses the first available model.
        """
        if not self.patient_features:
            self.compute_patient_features()
        
        data = []
        
        for patient_id in self.patient_results.keys():
            if patient_id not in self.patient_features:
                continue
            
            # Get model to analyze
            if model_name is None:
                target_model = list(self.patient_results[patient_id].keys())[0]
            elif model_name in self.patient_results[patient_id]:
                target_model = model_name
            else:
                continue
            
            # Combine features and performance metrics
            row = {'patient_id': patient_id}
            row.update(self.patient_features[patient_id])
            
            # Add performance metrics from pre-computed results
            performance_metrics = self.patient_results[patient_id][target_model]
            for key, value in performance_metrics.items():
                if key in ['y_true']:  # Skip glucose data arrays
                    continue
                elif isinstance(value, dict):
                    # Flatten nested dictionaries (clarke_zones, parkes_zones)
                    for sub_key, sub_value in value.items():
                        row[f"{key}_{sub_key}"] = sub_value
                else:
                    row[key] = value
            
            data.append(row)
        
        self.performance_data = pd.DataFrame(data)
        print(f"[OK] Created performance DataFrame with {len(self.performance_data)} patients and {len(self.performance_data.columns)} features")
        
        return self.performance_data
    
    def plot_patient_tsne(self, 
                          model_name: str = None,
                          save_path: str = None,
                          cluster_patients: bool = True,
                          use_pca_fallback: bool = True):
        """
        Create t-SNE visualization of patients colored by MAE performance.
        For small datasets, offers PCA as an alternative.
        
        Args:
            model_name: Model to analyze
            save_path: Path to save the plot
            cluster_patients: Whether to perform clustering analysis
            use_pca_fallback: Use PCA instead of t-SNE for very small datasets
        """
        print(f"\n{'='*60}")
        print("PATIENT t-SNE ANALYSIS")
        print(f"{'='*60}")
        
        if self.performance_data is None:
            self.create_performance_dataframe(model_name)
        
        # Use only performance metrics for t-SNE analysis
        feature_cols = ['mae', 'rmse', 'mape', 'mard']
        
        # Prepare data
        X = self.performance_data[feature_cols].values
        patient_ids = self.performance_data['patient_id'].values
        
        # Remove features with zero or very low variance
        feature_variances = np.var(X, axis=0)
        valid_features = feature_variances > 1e-6  # Remove essentially constant features
        X_filtered = X[:, valid_features]
        filtered_feature_cols = [feature_cols[i] for i in range(len(feature_cols)) if valid_features[i]]
        
        if np.sum(valid_features) < len(feature_cols):
            removed_count = len(feature_cols) - np.sum(valid_features)
            print(f"Removed {removed_count} features with zero/low variance")
        
        # Standardize features (important for distance-based methods)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filtered)
        
        print(f"Running dimensionality reduction on {len(patient_ids)} patients with {len(feature_cols)} performance metrics...")
        print(f"Using metrics: {', '.join(feature_cols)}")
        
        # Set appropriate perplexity for small datasets
        # Perplexity should be smaller than number of points and typically 5-50
        optimal_perplexity = min(max(2, len(patient_ids) // 4), 15)
        print(f"Using perplexity: {optimal_perplexity}")
        
        # For very small datasets, consider using PCA instead of t-SNE
        use_pca = use_pca_fallback and len(patient_ids) < 8
        
        if use_pca:
            print("Using PCA instead of t-SNE for small dataset")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            X_reduced = pca.fit_transform(X_scaled)
            method_name = "PCA"
            explained_variance = f"(Explained variance: {pca.explained_variance_ratio_[0]:.2f}, {pca.explained_variance_ratio_[1]:.2f})"
        else:
            # Run t-SNE with better parameters for small datasets
            tsne = TSNE(
                n_components=2, 
                random_state=42, 
                perplexity=optimal_perplexity,
                max_iter=1000,  # More iterations for better convergence
                learning_rate='auto',  # Adaptive learning rate
                init='pca',  # Better initialization
                metric='euclidean'
            )
            X_reduced = tsne.fit_transform(X_scaled)
            method_name = "t-SNE"
            explained_variance = ""
        
        # Calculate patient distances for proximity analysis
        from scipy.spatial.distance import pdist, squareform
        distances = pdist(X_scaled, metric='euclidean')
        distance_matrix = squareform(distances)
        
        # Find most similar patients for annotation
        np.fill_diagonal(distance_matrix, np.inf)
        most_similar_pairs = []
        for i in range(len(patient_ids)):
            closest_idx = np.argmin(distance_matrix[i, :])
            if distance_matrix[i, closest_idx] < np.inf:
                most_similar_pairs.append((i, closest_idx, distance_matrix[i, closest_idx]))
        
        # Sort by distance and get top 3 most similar pairs
        most_similar_pairs.sort(key=lambda x: x[2])
        top_similar_pairs = most_similar_pairs[:3]
        
        print(f"Top 3 most similar patient pairs:")
        for i, (idx1, idx2, dist) in enumerate(top_similar_pairs):
            pid1, pid2 = patient_ids[idx1], patient_ids[idx2]
            mae1 = self.performance_data.iloc[idx1]['mae']
            mae2 = self.performance_data.iloc[idx2]['mae']
            print(f"  {i+1}. Patients {pid1} & {pid2}: distance={dist:.3f}, MAE diff={abs(mae1-mae2):.3f}")
        
        # Restore diagonal for later use
        np.fill_diagonal(distance_matrix, 0)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Patient Performance Analysis: {self.experiment_name}\nModel: {model_name or "Default"} - {method_name} on Performance Metrics {explained_variance}', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Colored by MAE performance
        metric_values = self.performance_data['mae'].values
        
        scatter1 = axes[0, 0].scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                     c=metric_values, cmap='viridis_r', 
                                     s=100, alpha=0.7, edgecolors='black', linewidth=1)
        axes[0, 0].set_title('Patients in Performance Space\n(Colored by MAE)', 
                            fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel(f'{method_name} Component 1')
        axes[0, 0].set_ylabel(f'{method_name} Component 2')
        
        cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
        cbar1.set_label('MAE (mg/dL)')
        
        # Add patient ID labels
        for i, pid in enumerate(patient_ids):
            axes[0, 0].annotate(str(pid), (X_reduced[i, 0], X_reduced[i, 1]), 
                               fontsize=8, alpha=0.7)
        
        # Draw lines between most similar patients for proximity visualization
        for idx1, idx2, dist in top_similar_pairs:
            if dist < np.percentile([pair[2] for pair in most_similar_pairs], 33):  # Only show closest connections
                axes[0, 0].plot([X_reduced[idx1, 0], X_reduced[idx2, 0]], 
                               [X_reduced[idx1, 1], X_reduced[idx2, 1]], 
                               'gray', alpha=0.5, linewidth=1, linestyle='--')
        
        axes[0, 0].text(0.02, 0.98, f'Dashed lines: most similar patients \n(performance distance < {np.percentile([pair[2] for pair in most_similar_pairs], 33):.1f})', 
                       transform=axes[0, 0].transAxes, verticalalignment='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: Performance ranking
        performance_rank = stats.rankdata(metric_values, method='dense')
        
        scatter2 = axes[0, 1].scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                     c=performance_rank, cmap='RdYlGn_r', 
                                     s=100, alpha=0.7, edgecolors='black', linewidth=1)
        axes[0, 1].set_title('Performance Ranking\n(1=Best, Higher=Worse)', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel(f'{method_name} Component 1')
        axes[0, 1].set_ylabel(f'{method_name} Component 2')
        
        cbar2 = plt.colorbar(scatter2, ax=axes[0, 1])
        cbar2.set_label('Performance Rank')
        
        # Add patient ID labels
        for i, pid in enumerate(patient_ids):
            axes[0, 1].annotate(str(pid), (X_reduced[i, 0], X_reduced[i, 1]), 
                               fontsize=8, alpha=0.7)
        
        # Plot 3: Clustering analysis with detailed patient mapping
        if cluster_patients and len(patient_ids) >= 3:
            n_clusters = min(4, max(2, len(patient_ids) // 3))  # Better cluster number for small datasets
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)  # Use the same scaled and filtered data
            
            # Use distinct colors for clusters
            cluster_colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
            
            # Plot each cluster with distinct color
            for cluster_id in range(n_clusters):
                cluster_mask = clusters == cluster_id
                axes[1, 0].scatter(X_reduced[cluster_mask, 0], X_reduced[cluster_mask, 1], 
                                 c=[cluster_colors[cluster_id]], 
                                 s=150, alpha=0.7, edgecolors='black', linewidth=2,
                                 label=f'Cluster {cluster_id}', marker='o')
            
            axes[1, 0].set_title(f'Patient Clusters (K={n_clusters})\nColored by Cluster Assignment', 
                                fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel(f'{method_name} Component 1')
            axes[1, 0].set_ylabel(f'{method_name} Component 2')
            axes[1, 0].legend(loc='upper right', fontsize=8)
            axes[1, 0].grid(alpha=0.3)
            
            # Add patient ID labels with cluster-specific colors
            for i, pid in enumerate(patient_ids):
                cluster_id = clusters[i]
                axes[1, 0].annotate(str(pid), (X_reduced[i, 0], X_reduced[i, 1]), 
                                   fontsize=9, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', 
                                           facecolor=cluster_colors[cluster_id], 
                                           edgecolor='black', 
                                           alpha=0.8))
            
            # Analyze cluster performance
            cluster_analysis = []
            for cluster_id in range(n_clusters):
                cluster_mask = clusters == cluster_id
                cluster_patients = patient_ids[cluster_mask]
                cluster_performance = metric_values[cluster_mask]
                
                cluster_analysis.append({
                    'cluster': cluster_id,
                    'n_patients': len(cluster_patients),
                    'patients': cluster_patients.tolist(),
                    'mean_performance': np.mean(cluster_performance),
                    'std_performance': np.std(cluster_performance),
                    'min_performance': np.min(cluster_performance),
                    'max_performance': np.max(cluster_performance)
                })
            
            # Print detailed cluster analysis to console
            print(f"\n{'='*60}")
            print(f"CLUSTER ASSIGNMENT ANALYSIS (K={n_clusters})")
            print(f"{'='*60}")
            for ca in sorted(cluster_analysis, key=lambda x: x['mean_performance']):
                print(f"\nCluster {ca['cluster']} - {ca['n_patients']} patients")
                print(f"  Patient IDs: {ca['patients']}")
                print(f"  MAE: {ca['mean_performance']:.3f} ± {ca['std_performance']:.3f} mg/dL")
                print(f"  Range: [{ca['min_performance']:.3f}, {ca['max_performance']:.3f}] mg/dL")
                
                # Characterize cluster
                if ca['mean_performance'] < np.percentile(metric_values, 33):
                    performance_label = "HIGH PERFORMANCE (Easy to predict)"
                elif ca['mean_performance'] < np.percentile(metric_values, 67):
                    performance_label = "MEDIUM PERFORMANCE"
                else:
                    performance_label = "LOW PERFORMANCE (Hard to predict)"
                print(f"  Classification: {performance_label}")
            
            # Store cluster assignments in performance_data
            self.performance_data['kmeans_cluster'] = clusters
            
            # Display cluster analysis on plot with clearer formatting
            cluster_text = f"Performance by Cluster:\n"
            cluster_text += "-" * 35 + "\n"
            for ca in sorted(cluster_analysis, key=lambda x: x['mean_performance']):
                cluster_text += f"Cluster {ca['cluster']}: n={ca['n_patients']}\n"
                cluster_text += f"  MAE: {ca['mean_performance']:.2f}±{ca['std_performance']:.2f}\n"
                cluster_text += f"  Patients: {ca['patients']}\n"
            
            axes[1, 0].text(0.02, 0.98, cluster_text, transform=axes[1, 0].transAxes,
                            verticalalignment='top', fontsize=7, family='monospace',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='black'))
        else:
            axes[1, 0].text(0.5, 0.5, 'Not enough patients\nfor clustering analysis', 
                           transform=axes[1, 0].transAxes, ha='center', va='center',
                           fontsize=12, style='italic')
            axes[1, 0].set_xticks([])
            axes[1, 0].set_yticks([])
        
        # Plot 4: Patient feature correlations with MAE (using all patient features)
        # Get all patient feature columns for correlation analysis
        all_feature_cols = [col for col in self.performance_data.columns 
                           if col not in ['patient_id', 'mae', 'rmse', 'mape', 'mard', 'tir', 
                                         'hypo_events', 'hyper_events', 'clarke_a_b', 'parkes_a_b', 
                                         'n_predictions', 'performance_group'] and not col.startswith('clarke_zones_') and not col.startswith('parkes_zones_')]
        
        # Filter out low-variance features for better analysis
        X_feature_analysis = self.performance_data[all_feature_cols].values
        feature_variances = np.var(X_feature_analysis, axis=0)
        valid_feature_mask = feature_variances > 1e-6
        filtered_all_feature_cols = [all_feature_cols[i] for i in range(len(all_feature_cols)) if valid_feature_mask[i]]
        
        feature_correlations = []
        for feature in filtered_all_feature_cols:  # Use all filtered patient features
            corr, p_value = stats.pearsonr(self.performance_data[feature], metric_values)
            feature_correlations.append({
                'feature': feature,
                'correlation': corr,
                'p_value': p_value,
                'abs_correlation': abs(corr)
            })
        
        # Sort by absolute correlation
        feature_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        # Plot top correlations
        top_features = feature_correlations[:10]
        feature_names = [f['feature'] for f in top_features]
        correlations = [f['correlation'] for f in top_features]
        
        bars = axes[1, 1].barh(range(len(feature_names)), correlations, 
                              color=['red' if c < 0 else 'green' for c in correlations])
        axes[1, 1].set_yticks(range(len(feature_names)))
        axes[1, 1].set_yticklabels(feature_names, fontsize=8)
        axes[1, 1].set_xlabel('Correlation with MAE')
        axes[1, 1].set_title('Top Patient Feature Correlations\n(with MAE)', fontsize=14, fontweight='bold')
        axes[1, 1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add correlation values on bars
        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            axes[1, 1].text(corr + (0.01 if corr > 0 else -0.01), i, 
                            f'{corr:.3f}', va='center', 
                            ha='left' if corr > 0 else 'right', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Patient t-SNE plot saved to {save_path}")
            plt.close()
        else:
            plt.show()
        
        print(f"[OK] Patient t-SNE analysis complete")
        
        return feature_correlations
    
    def plot_glucose_tsne(self, 
                         model_name: str = None,
                         save_path: str = None,
                         max_samples: int = 1000,
                         use_pca_fallback: bool = True):
        """
        Create t-SNE visualization of actual glucose values for each patient,
        colored by their MAE performance.
        
        Args:
            model_name: Model to analyze
            save_path: Path to save the plot
            max_samples: Maximum number of glucose samples per patient to use
            use_pca_fallback: Use PCA instead of t-SNE for very small datasets
        """
        print(f"\n{'='*60}")
        print("GLUCOSE VALUES t-SNE ANALYSIS")
        print(f"{'='*60}")
        
        if not self.patient_results:
            self.load_experiment_results()
        
        # Prepare glucose data for each patient
        glucose_data = []
        patient_labels = []
        mae_values = []
        
        for patient_id in self.patient_results.keys():
            # Get model to analyze
            if model_name is None:
                target_model = list(self.patient_results[patient_id].keys())[0]
            elif model_name in self.patient_results[patient_id]:
                target_model = model_name
            else:
                continue
            
            y_true = self.patient_results[patient_id][target_model]['y_true']
            mae = self.patient_results[patient_id][target_model]['mae']
            
            if y_true is None or len(y_true) == 0:
                print(f"    Warning: No glucose data available for patient {patient_id}")
                continue
            
            # Sample glucose values if too many
            if len(y_true) > max_samples:
                indices = np.random.choice(len(y_true), max_samples, replace=False)
                sampled_glucose = y_true[indices]
            else:
                sampled_glucose = y_true
            
            # Add glucose values
            glucose_data.extend(sampled_glucose)
            patient_labels.extend([patient_id] * len(sampled_glucose))
            mae_values.extend([mae] * len(sampled_glucose))
        
        if len(glucose_data) == 0:
            print("    Error: No glucose data available for analysis")
            return None
        
        # Convert to numpy arrays
        glucose_array = np.array(glucose_data).reshape(-1, 1)
        patient_array = np.array(patient_labels)
        mae_array = np.array(mae_values)
        
        print(f"Analyzing {len(glucose_data)} glucose values from {len(set(patient_labels))} patients...")
        
        # Standardize glucose values
        scaler = StandardScaler()
        glucose_scaled = scaler.fit_transform(glucose_array)
        
        # For very small datasets, use PCA instead of t-SNE
        n_patients = len(set(patient_labels))
        use_pca = use_pca_fallback and n_patients < 8
        
        if use_pca:
            print("Using PCA instead of t-SNE for small dataset")
            from sklearn.decomposition import PCA
            # For 1D data, we'll add some noise to create a 2D embedding
            glucose_with_noise = np.column_stack([glucose_scaled.flatten(), 
                                                 np.random.normal(0, 0.1, len(glucose_scaled))])
            pca = PCA(n_components=2, random_state=42)
            glucose_reduced = pca.fit_transform(glucose_with_noise)
            method_name = "PCA"
            explained_variance = f"(Explained variance: {pca.explained_variance_ratio_[0]:.2f}, {pca.explained_variance_ratio_[1]:.2f})"
        else:
            # Use t-SNE with appropriate perplexity
            optimal_perplexity = min(max(5, len(glucose_data) // 100), 50)
            print(f"Using t-SNE with perplexity: {optimal_perplexity}")
            
            # For 1D glucose data, add time-based feature for better embedding
            time_feature = np.arange(len(glucose_data)) / len(glucose_data)
            glucose_with_time = np.column_stack([glucose_scaled.flatten(), time_feature]) # Add time as a feature
            
            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=optimal_perplexity,
                max_iter=1000,
                learning_rate='auto',
                init='pca',
                metric='euclidean'
            )
            glucose_reduced = tsne.fit_transform(glucose_with_time)
            method_name = "t-SNE"
            explained_variance = ""
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Glucose Values Analysis: {self.experiment_name}\nModel: {model_name or "Default"} - {method_name} {explained_variance}', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Colored by MAE
        scatter1 = axes[0].scatter(glucose_reduced[:, 0], glucose_reduced[:, 1], 
                                  c=mae_array, cmap='viridis_r', 
                                  s=20, alpha=0.6, edgecolors='none')
        axes[0].set_title('Glucose Values colored by Patient MAE', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel(f'{method_name} Component 1')
        axes[0].set_ylabel(f'{method_name} Component 2')
        
        cbar1 = plt.colorbar(scatter1, ax=axes[0])
        cbar1.set_label('MAE (mg/dL)')
        
        # Plot 2: Colored by patient ID with centroids
        unique_patients = sorted(set(patient_labels))
        patient_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_patients)))
        
        for i, patient_id in enumerate(unique_patients):
            patient_mask = patient_array == patient_id
            patient_points = glucose_reduced[patient_mask]
            patient_mae = mae_array[patient_mask][0]  # All points have same MAE
            
            # Plot patient points
            axes[1].scatter(patient_points[:, 0], patient_points[:, 1], 
                           c=[patient_colors[i]], s=20, alpha=0.6, 
                           label=f'P{patient_id} (MAE:{patient_mae:.2f})')
        
        axes[1].set_title('Glucose Values by Patient', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel(f'{method_name} Component 1')
        axes[1].set_ylabel(f'{method_name} Component 2')
        
        # Add legend with patient ID and color mapping
        if len(unique_patients) <= 15:
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, 
                          framealpha=0.9, title='Patient ID (MAE)')
        else:
            # For many patients, use compact legend with multiple columns
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, 
                          ncol=2, framealpha=0.9, title='Patient ID (MAE)')
        
        # Calculate patient similarity based on glucose distributions
        print(f"\nPatient Glucose Distribution Analysis:")
        print("-" * 50)
        
        patient_stats = {}
        for patient_id in unique_patients:
            patient_mask = patient_array == patient_id
            patient_glucose = glucose_array[patient_mask].flatten()
            patient_mae = mae_array[patient_mask][0]
            
            patient_stats[patient_id] = {
                'mean_glucose': np.mean(patient_glucose),
                'std_glucose': np.std(patient_glucose),
                'min_glucose': np.min(patient_glucose),
                'max_glucose': np.max(patient_glucose),
                'mae': patient_mae,
                'n_samples': len(patient_glucose)
            }
            
            print(f"Patient {patient_id}: Mean={patient_stats[patient_id]['mean_glucose']:.1f}, "
                  f"Std={patient_stats[patient_id]['std_glucose']:.1f}, "
                  f"Range=[{patient_stats[patient_id]['min_glucose']:.0f}-{patient_stats[patient_id]['max_glucose']:.0f}], "
                  f"MAE={patient_stats[patient_id]['mae']:.3f}, "
                  f"Samples={patient_stats[patient_id]['n_samples']}")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n[OK] Glucose t-SNE plot saved to {save_path}")
            plt.close()
        else:
            plt.show()
        
        print(f"[OK] Glucose t-SNE analysis complete")
        
        return patient_stats
    
    
    def plot_cluster_tsne_correlation(self,
                                      model_name: str = None,
                                      output_dir: str = 'patient_analysis_results',
                                      save_path: str = None,
                                      method: str = 't-SNE',
                                      cluster_method: str = 'kmeans',
                                      n_clusters: int = None):
        """
        Create detailed correlation plot between t-SNE positions and cluster assignments.
        This function shows:
        1. Side-by-side comparison of performance-based coloring vs cluster assignment
        2. Cluster hulls/boundaries to show spatial groupings
        3. Performance pr ofiles for each cluster
        4. Patient ID mappings with cross-references
        
        Parameters:
        -----------
        model_name : str, optional
            Model name to analyze (RNN, LSTM, GRU)
        output_dir : str
            Output directory for plots (default: 'patient_analysis_results')
        save_path : str, optional
            Path to save plot (default: auto-generated)
        method : str
            Dimensionality reduction method ('t-SNE', 'PCA', 'UMAP')
        cluster_method : str
            Clustering method ('kmeans', 'performance')
        n_clusters : int, optional
            Number of clusters (auto-determined if None)
        
        Returns:
        --------
        dict : Detailed cluster information with patient assignments
        """
        from scipy.spatial import ConvexHull
        from matplotlib.patches import Polygon
        
        print(f"\n{'='*60}")
        print(f"Creating Cluster-t-SNE Correlation Analysis")
        print(f"{'='*60}")
        
        # Load data
        if self.performance_data is None or self.features_data is None:
            if model_name:
                # Data will be loaded from existing analyzer state
                if self.performance_data is None:
                    print(f"[ERROR] No data loaded. Please load experiment results first.")
                    return None
            else:
                # Try to find available model from existing data
                if self.performance_data is None:
                    print(f"[ERROR] No experiment results loaded")
                    return None
                model_name = 'GRU'  # Default
        
        if self.performance_data is None:
            print("[ERROR] Failed to load data")
            return None
        
        # Use MAE as primary metric
        metric = 'mae'
        valid_mask = self.performance_data[metric].notna()
        patient_ids = self.performance_data.loc[valid_mask, 'patient_id'].values
        metric_values = self.performance_data.loc[valid_mask, metric].values
        
        if len(patient_ids) < 3:
            print(f"[WARNING] Need at least 3 patients, found {len(patient_ids)}")
            return None
        
        # Get features and scale
        feature_cols = [col for col in self.performance_data.columns 
                       if col not in ['patient_id', 'mae', 'rmse', 'mape']]
        X = self.performance_data.loc[valid_mask, feature_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply dimensionality reduction
        if method.upper() == 'T-SNE':
            from sklearn.manifold import TSNE
            perplexity = 5  # Fixed perplexity
            effective_perplexity = min(perplexity, (len(patient_ids) - 1) // 2)
            tsne = TSNE(n_components=2, random_state=42, perplexity=effective_perplexity, max_iter=1000)
            X_reduced = tsne.fit_transform(X_scaled)
            method_name = 't-SNE'
        elif method.upper() == 'PCA':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            X_reduced = pca.fit_transform(X_scaled)
            method_name = 'PCA'
        elif method.upper() == 'UMAP':
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            X_reduced = reducer.fit_transform(X_scaled)
            method_name = 'UMAP'
        else:
            print(f"[WARNING] Unknown method '{method}', using PCA")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            X_reduced = pca.fit_transform(X_scaled)
            method_name = 'PCA'
        
        # Determine clustering
        if n_clusters is None:
            n_clusters = min(4, max(2, len(patient_ids) // 3))
        
        if cluster_method == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
        else:  # performance-based clustering
            # Create tertile-based clusters
            performance_percentiles = [np.percentile(metric_values, p) for p in [33, 67]]
            clusters = np.digitize(metric_values, performance_percentiles)
            n_clusters = len(np.unique(clusters))
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(f'Cluster-{method_name} Correlation Analysis: {model_name}\n'
                     f'{n_clusters} Clusters ({cluster_method})', 
                     fontsize=16, fontweight='bold')
        
        # Define colors
        cluster_colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        performance_cmap = plt.cm.RdYlGn_r
        
        # Plot 1: Performance-based coloring
        scatter1 = axes[0, 0].scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                     c=metric_values, cmap=performance_cmap,
                                     s=200, alpha=0.7, edgecolors='black', linewidth=2)
        axes[0, 0].set_title(f'{method_name} Colored by MAE Performance', 
                            fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel(f'{method_name} Component 1')
        axes[0, 0].set_ylabel(f'{method_name} Component 2')
        axes[0, 0].grid(alpha=0.3)
        
        # Add patient labels
        for i, pid in enumerate(patient_ids):
            axes[0, 0].annotate(str(pid), (X_reduced[i, 0], X_reduced[i, 1]), 
                               fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                       edgecolor='black', alpha=0.7))
        
        cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
        cbar1.set_label('MAE (mg/dL)', fontsize=10)
        
        # Plot 2: Cluster-based coloring with hulls
        for cluster_id in range(n_clusters):
            cluster_mask = clusters == cluster_id
            axes[0, 1].scatter(X_reduced[cluster_mask, 0], X_reduced[cluster_mask, 1], 
                             c=[cluster_colors[cluster_id]], 
                             s=200, alpha=0.7, edgecolors='black', linewidth=2,
                             label=f'Cluster {cluster_id}', marker='o')
            
            # Draw convex hull if cluster has 3+ points
            cluster_points = X_reduced[cluster_mask]
            if len(cluster_points) >= 3:
                try:
                    hull = ConvexHull(cluster_points)
                    for simplex in hull.simplices:
                        axes[0, 1].plot(cluster_points[simplex, 0], 
                                      cluster_points[simplex, 1], 
                                      color=cluster_colors[cluster_id], 
                                      linestyle='--', linewidth=2, alpha=0.5)
                    
                    # Fill hull
                    hull_points = cluster_points[hull.vertices]
                    polygon = Polygon(hull_points, facecolor=cluster_colors[cluster_id], 
                                    alpha=0.15, edgecolor=cluster_colors[cluster_id], 
                                    linewidth=2, linestyle='--')
                    axes[0, 1].add_patch(polygon)
                except:
                    pass  # Skip if hull computation fails
        
        axes[0, 1].set_title(f'{method_name} Colored by Cluster Assignment\n(with Cluster Boundaries)', 
                            fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel(f'{method_name} Component 1')
        axes[0, 1].set_ylabel(f'{method_name} Component 2')
        axes[0, 1].legend(loc='upper right', fontsize=9)
        axes[0, 1].grid(alpha=0.3)
        
        # Add patient labels with cluster colors
        for i, pid in enumerate(patient_ids):
            cluster_id = clusters[i]
            axes[0, 1].annotate(str(pid), (X_reduced[i, 0], X_reduced[i, 1]), 
                               fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor=cluster_colors[cluster_id], 
                                       edgecolor='black', alpha=0.8))
        
        # Plot 3: Cluster performance profiles
        cluster_info = []
        for cluster_id in range(n_clusters):
            cluster_mask = clusters == cluster_id
            cluster_patients = patient_ids[cluster_mask]
            cluster_mae = metric_values[cluster_mask]
            
            cluster_info.append({
                'cluster': cluster_id,
                'n_patients': len(cluster_patients),
                'patients': cluster_patients.tolist(),
                'mean_mae': np.mean(cluster_mae),
                'std_mae': np.std(cluster_mae),
                'min_mae': np.min(cluster_mae),
                'max_mae': np.max(cluster_mae),
                'median_mae': np.median(cluster_mae)
            })
        
        # Bar plot of cluster performance
        cluster_ids_plot = [ci['cluster'] for ci in cluster_info]
        mean_maes = [ci['mean_mae'] for ci in cluster_info]
        std_maes = [ci['std_mae'] for ci in cluster_info]
        
        bars = axes[1, 0].bar(cluster_ids_plot, mean_maes, yerr=std_maes, 
                             color=cluster_colors, alpha=0.7, edgecolor='black', 
                             linewidth=2, capsize=5)
        axes[1, 0].set_title('Performance Profile by Cluster', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Cluster ID', fontsize=10)
        axes[1, 0].set_ylabel('Mean MAE (mg/dL)', fontsize=10)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, ci in zip(bars, cluster_info):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{ci["mean_mae"]:.2f}±{ci["std_mae"]:.2f}\nn={ci["n_patients"]}',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Plot 4: Patient-cluster mapping table
        axes[1, 1].axis('off')
        
        # Create detailed mapping text
        mapping_text = "PATIENT-CLUSTER MAPPING\n"
        mapping_text += "=" * 55 + "\n\n"
        
        for ci in sorted(cluster_info, key=lambda x: x['mean_mae']):
            # Classify performance
            if ci['mean_mae'] < np.percentile(metric_values, 33):
                perf_label = "HIGH PERFORMANCE"
                perf_emoji = "🟢"
            elif ci['mean_mae'] < np.percentile(metric_values, 67):
                perf_label = "MEDIUM PERFORMANCE"
                perf_emoji = "🟡"
            else:
                perf_label = "LOW PERFORMANCE"
                perf_emoji = "🔴"
            
            mapping_text += f"{perf_emoji} Cluster {ci['cluster']} - {perf_label}\n"
            mapping_text += f"{'─' * 53}\n"
            mapping_text += f"Size: {ci['n_patients']} patients\n"
            mapping_text += f"Patient IDs: {ci['patients']}\n"
            mapping_text += f"MAE: {ci['mean_mae']:.3f} ± {ci['std_mae']:.3f} mg/dL\n"
            mapping_text += f"Range: [{ci['min_mae']:.3f}, {ci['max_mae']:.3f}] mg/dL\n"
            mapping_text += f"Median: {ci['median_mae']:.3f} mg/dL\n\n"
        
        # Add spatial clustering quality metric
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        try:
            silhouette = silhouette_score(X_reduced, clusters)
            davies_bouldin = davies_bouldin_score(X_reduced, clusters)
            mapping_text += f"\nSpatial Clustering Quality:\n"
            mapping_text += f"{'─' * 53}\n"
            mapping_text += f"Silhouette Score: {silhouette:.3f}\n"
            mapping_text += f"  (Range: [-1, 1], higher = better separation)\n"
            mapping_text += f"Davies-Bouldin Index: {davies_bouldin:.3f}\n"
            mapping_text += f"  (Range: [0, ∞], lower = better separation)\n"
        except:
            pass
        
        axes[1, 1].text(0.05, 0.95, mapping_text, transform=axes[1, 1].transAxes,
                       verticalalignment='top', fontsize=9, family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', 
                               alpha=0.9, edgecolor='black', linewidth=2))
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, 
                                    f'cluster_tsne_correlation_{model_name}_{method.lower()}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[OK] Cluster-t-SNE correlation plot saved to {save_path}")
        plt.close()
        
        # Print detailed console output
        print(f"\n{'='*60}")
        print(f"DETAILED CLUSTER ANALYSIS")
        print(f"{'='*60}")
        print(f"Method: {method_name} ({cluster_method} clustering)")
        print(f"Number of clusters: {n_clusters}")
        print(f"Total patients: {len(patient_ids)}")
        
        for ci in sorted(cluster_info, key=lambda x: x['mean_mae']):
            print(f"\n{'-'*60}")
            print(f"Cluster {ci['cluster']}:")
            print(f"  Size: {ci['n_patients']} patients")
            print(f"  Patients: {ci['patients']}")
            print(f"  Performance: {ci['mean_mae']:.3f} ± {ci['std_mae']:.3f} mg/dL")
            print(f"  Range: [{ci['min_mae']:.3f}, {ci['max_mae']:.3f}]")
            print(f"  Median: {ci['median_mae']:.3f} mg/dL")
            
            # Show individual patient details
            for pid in ci['patients']:
                pid_idx = np.where(patient_ids == pid)[0][0]
                print(f"    Patient {pid}: MAE = {metric_values[pid_idx]:.3f} mg/dL, "
                      f"Position = ({X_reduced[pid_idx, 0]:.2f}, {X_reduced[pid_idx, 1]:.2f})")
        
        print(f"\n{'='*60}")
        print(f"[OK] Cluster-t-SNE correlation analysis complete")
        
        return {
            'cluster_info': cluster_info,
            'clusters': clusters,
            'patient_ids': patient_ids,
            'tsne_positions': X_reduced,
            'n_clusters': n_clusters,
            'method': method_name
        }
    
    def plot_train_test_glucose_comparison(self,
                                          data_root: str = 'data',
                                          model_name: str = None,
                                          save_path: str = None,
                                          max_samples: int = 1000,
                                          use_pca_fallback: bool = True):
        """
        Create 4-panel comparison of glucose t-SNE for training and test data.
        
        Creates:
        - Plot 1 (top-left): Training data colored by patient ID
        - Plot 2 (top-right): Training data colored by mean MAE
        - Plot 3 (bottom-left): Test data colored by patient ID
        - Plot 4 (bottom-right): Test data colored by mean MAE
        
        Args:
            data_root: Root directory containing raw data (default: 'data')
            model_name: Model to analyze for MAE values
            save_path: Path to save the plot
            max_samples: Maximum number of glucose samples per patient to use
            use_pca_fallback: Use PCA instead of t-SNE for very small datasets
        """
        print(f"\n{'='*60}")
        print("TRAIN/TEST GLUCOSE COMPARISON t-SNE ANALYSIS")
        print(f"{'='*60}")
        
        if not self.patient_results:
            self.load_experiment_results()
        
        # Import data loading functions
        import sys
        from pathlib import Path
        sys.path.append(str(Path(data_root).parent / 'benchmark'))
        from benchmark.data.loaders import OhioT1DMDataLoader
        
        # Initialize data loader
        loader = OhioT1DMDataLoader(data_dir=data_root, version=['2018', '2020'])
        
        # Prepare data for both train and test
        datasets = {'train': {}, 'test': {}}
        patient_ids = sorted(self.patient_results.keys())
        
        print(f"Loading data for {len(patient_ids)} patients...")
        
        for patient_id in patient_ids:
            try:
                # Load training data
                train_df = loader.load_patient_data(patient_id, mode='train')
                datasets['train'][patient_id] = train_df
                
                # Load test data
                test_df = loader.load_patient_data(patient_id, mode='test')
                datasets['test'][patient_id] = test_df
                
                print(f"    Loaded patient {patient_id}: train={len(train_df)} samples, test={len(test_df)} samples")
                    
            except Exception as e:
                print(f"    Error loading patient {patient_id}: {e}")
                continue
        
        # Get MAE values for each patient
        if model_name is None:
            model_name = list(self.patient_results[patient_ids[0]].keys())[0]
        
        patient_mae = {}
        for patient_id in patient_ids:
            if model_name in self.patient_results[patient_id]:
                patient_mae[patient_id] = self.patient_results[patient_id][model_name]['mae']
            else:
                patient_mae[patient_id] = np.nan
        
        # Prepare glucose data for both datasets
        plot_data = {}
        for dataset_type in ['train', 'test']:
            glucose_data = []
            patient_labels = []
            mae_values = []
            
            for patient_id in patient_ids:
                if patient_id not in datasets[dataset_type]:
                    continue
                
                df = datasets[dataset_type][patient_id]
                
                if 'glucose' not in df.columns:
                    print(f"    Warning: No glucose column for patient {patient_id} in {dataset_type}")
                    continue
                
                # Extract glucose values and convert to float
                glucose_values = pd.to_numeric(df['glucose'], errors='coerce').values
                glucose_values = glucose_values[~np.isnan(glucose_values)]  # Remove NaN
                
                if len(glucose_values) == 0:
                    continue
                
                # Sample if too many
                if len(glucose_values) > max_samples:
                    indices = np.random.choice(len(glucose_values), max_samples, replace=False)
                    sampled_glucose = glucose_values[indices]
                else:
                    sampled_glucose = glucose_values
                
                glucose_data.extend(sampled_glucose)
                patient_labels.extend([patient_id] * len(sampled_glucose))
                mae_values.extend([patient_mae[patient_id]] * len(sampled_glucose))
            
            plot_data[dataset_type] = {
                'glucose': np.array(glucose_data),
                'patients': np.array(patient_labels),
                'mae': np.array(mae_values)
            }
            
            print(f"{dataset_type.capitalize()}: {len(glucose_data)} glucose values from {len(set(patient_labels))} patients")
        
        # Create t-SNE/PCA embeddings for both datasets
        n_patients = len(patient_ids)
        use_pca = use_pca_fallback and n_patients < 8
        method_name = "PCA" if use_pca else "t-SNE"
        
        embeddings = {}
        for dataset_type in ['train', 'test']:
            glucose_array = plot_data[dataset_type]['glucose'].reshape(-1, 1)
            
            # Standardize
            scaler = StandardScaler()
            glucose_scaled = scaler.fit_transform(glucose_array)
            
            if use_pca:
                from sklearn.decomposition import PCA
                glucose_with_noise = np.column_stack([
                    glucose_scaled.flatten(), 
                    np.random.normal(0, 0.1, len(glucose_scaled))
                ])
                pca = PCA(n_components=2, random_state=42)
                embedding = pca.fit_transform(glucose_with_noise)
                explained_var = f"(Var: {pca.explained_variance_ratio_[0]:.2f}, {pca.explained_variance_ratio_[1]:.2f})"
            else:
                optimal_perplexity = min(max(5, len(glucose_array) // 100), 50)
                time_feature = np.arange(len(glucose_array)) / len(glucose_array)
                glucose_with_time = np.column_stack([glucose_scaled.flatten(), time_feature])
                
                tsne = TSNE(
                    n_components=2,
                    random_state=42,
                    perplexity=optimal_perplexity,
                    max_iter=1000,
                    learning_rate='auto',
                    init='pca',
                    metric='euclidean'
                )
                embedding = tsne.fit_transform(glucose_with_time)
                explained_var = ""
            
            embeddings[dataset_type] = embedding
        
        # Create 2x2 plot
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'Train vs Test Glucose Analysis: {self.experiment_name}\n'
                    f'Model: {model_name} - {method_name} {explained_var}',
                    fontsize=18, fontweight='bold', y=0.995)
        
        unique_patients = sorted(set(plot_data['train']['patients']))
        patient_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_patients)))
        
        # Plot each panel
        for row, dataset_type in enumerate(['train', 'test']):
            data = plot_data[dataset_type]
            embedding = embeddings[dataset_type]
            
            # Left column: Colored by patient ID
            ax_patient = axes[row, 0]
            for i, patient_id in enumerate(unique_patients):
                if patient_id not in data['patients']:
                    continue
                patient_mask = data['patients'] == patient_id
                patient_points = embedding[patient_mask]
                
                ax_patient.scatter(
                    patient_points[:, 0], patient_points[:, 1],
                    c=[patient_colors[i]], s=20, alpha=0.6,
                    label=f'P{patient_id}'
                )
            
            ax_patient.set_title(f'{dataset_type.capitalize()} Data - Colored by Patient',
                               fontsize=14, fontweight='bold')
            ax_patient.set_xlabel(f'{method_name} Component 1')
            ax_patient.set_ylabel(f'{method_name} Component 2')
            
            # Add legend with patient ID and color mapping
            if len(unique_patients) <= 15:
                ax_patient.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8,
                                framealpha=0.9, title='Patient ID')
            else:
                # For many patients, use compact legend with multiple columns
                ax_patient.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7,
                                ncol=2, framealpha=0.9, title='Patient ID')
            
            # Right column: Colored by MAE
            ax_mae = axes[row, 1]
            scatter = ax_mae.scatter(
                embedding[:, 0], embedding[:, 1],
                c=data['mae'], cmap='viridis_r',
                s=20, alpha=0.6, edgecolors='none'
            )
            
            ax_mae.set_title(f'{dataset_type.capitalize()} Data - Colored by Patient MAE',
                           fontsize=14, fontweight='bold')
            ax_mae.set_xlabel(f'{method_name} Component 1')
            ax_mae.set_ylabel(f'{method_name} Component 2')
            
            cbar = plt.colorbar(scatter, ax=ax_mae)
            cbar.set_label('MAE (mg/dL)', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n[OK] Train/test comparison plot saved to {save_path}")
            plt.close()
        else:
            plt.show()
        
        print(f"[OK] Train/test glucose comparison complete")
        
        # Print statistics
        print(f"\n{'='*60}")
        print("DATASET STATISTICS")
        print(f"{'='*60}")
        for dataset_type in ['train', 'test']:
            data = plot_data[dataset_type]
            print(f"\n{dataset_type.capitalize()} Data:")
            print(f"  Total samples: {len(data['glucose'])}")
            print(f"  Glucose range: [{np.min(data['glucose']):.1f}, {np.max(data['glucose']):.1f}] mg/dL")
            print(f"  Glucose mean: {np.mean(data['glucose']):.1f} ± {np.std(data['glucose']):.1f} mg/dL")
            print(f"  Number of patients: {len(set(data['patients']))}")
        
        return plot_data
    
    def quantify_clusters_statistical_analysis(self, 
                                              model_name: str = None,
                                              save_path: str = None,
                                              n_clusters: int = 3):
        """
        Comprehensive statistical analysis to quantify cluster differences.
        
        Performs:
        1. Cluster-wise feature distribution comparison (boxplots)
        2. ANOVA/Kruskal-Wallis tests for each feature
        3. Post-hoc pairwise comparisons
        4. Effect size calculations (Cohen's d)
        
        Args:
            model_name: Model to analyze
            save_path: Path to save statistical report
            n_clusters: Number of clusters to create
            
        Returns:
            Dictionary with statistical test results
        """
        print(f"\n{'='*60}")
        print("CLUSTER QUANTIFICATION - STATISTICAL ANALYSIS")
        print(f"{'='*60}")
        
        if self.performance_data is None:
            self.create_performance_dataframe(model_name)
        
        # Perform clustering if not already done
        if 'performance_group' not in self.performance_data.columns:
            metric_values = self.performance_data['mae'].values
            low_thresh = np.percentile(metric_values, 33.33)
            high_thresh = np.percentile(metric_values, 66.67)
            
            performance_groups = []
            for val in metric_values:
                if val <= low_thresh:
                    performance_groups.append('High_Performance')
                elif val <= high_thresh:
                    performance_groups.append('Medium_Performance')
                else:
                    performance_groups.append('Low_Performance')
            
            self.performance_data['performance_group'] = performance_groups
        
        # Get feature columns
        feature_cols = [col for col in self.performance_data.columns 
                       if col not in ['patient_id', 'mae', 'rmse', 'mape', 'mard', 'tir', 
                                     'hypo_events', 'hyper_events', 'clarke_a_b', 'parkes_a_b',
                                     'n_predictions', 'performance_group'] 
                       and not col.startswith('clarke_zones_') 
                       and not col.startswith('parkes_zones_')]
        
        # Filter low-variance features
        X_analysis = self.performance_data[feature_cols].values
        feature_variances = np.var(X_analysis, axis=0)
        valid_features = feature_variances > 1e-6
        feature_cols = [feature_cols[i] for i in range(len(feature_cols)) if valid_features[i]]
        
        print(f"Analyzing {len(feature_cols)} features across clusters...")
        
        # Storage for results
        statistical_results = {
            'features': [],
            'anova_results': [],
            'pairwise_results': [],
            'effect_sizes': []
        }
        
        # Perform ANOVA/Kruskal-Wallis for each feature
        groups = ['High_Performance', 'Medium_Performance', 'Low_Performance']
        
        print(f"\n{'='*60}")
        print("STATISTICAL TESTS FOR EACH FEATURE")
        print(f"{'='*60}\n")
        
        for feature in feature_cols:
            # Get data for each group
            group_data = {}
            for group in groups:
                mask = self.performance_data['performance_group'] == group
                group_data[group] = self.performance_data.loc[mask, feature].values
            
            # Check normality (Shapiro-Wilk test)
            normality_pvalues = []
            for group, data in group_data.items():
                if len(data) >= 3:
                    from scipy.stats import shapiro
                    _, p = shapiro(data)
                    normality_pvalues.append(p)
            
            # Use parametric (ANOVA) if all groups normal, else non-parametric (Kruskal-Wallis)
            use_parametric = all(p > 0.05 for p in normality_pvalues) if normality_pvalues else False
            
            if use_parametric:
                # One-way ANOVA
                from scipy.stats import f_oneway
                stat, p_value = f_oneway(*group_data.values())
                test_name = "ANOVA"
            else:
                # Kruskal-Wallis H-test
                from scipy.stats import kruskal
                stat, p_value = kruskal(*group_data.values())
                test_name = "Kruskal-Wallis"
            
            # Store results
            result = {
                'feature': feature,
                'test': test_name,
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'group_means': {g: np.mean(d) for g, d in group_data.items()},
                'group_stds': {g: np.std(d) for g, d in group_data.items()}
            }
            
            statistical_results['anova_results'].append(result)
            
            # Post-hoc pairwise comparisons if significant
            if p_value < 0.05:
                from scipy.stats import mannwhitneyu, ttest_ind
                
                pairwise = []
                for i, g1 in enumerate(groups):
                    for g2 in groups[i+1:]:
                        if use_parametric:
                            stat_pw, p_pw = ttest_ind(group_data[g1], group_data[g2])
                            test_pw = "t-test"
                        else:
                            stat_pw, p_pw = mannwhitneyu(group_data[g1], group_data[g2])
                            test_pw = "Mann-Whitney U"
                        
                        # Calculate Cohen's d effect size
                        mean1, mean2 = np.mean(group_data[g1]), np.mean(group_data[g2])
                        std1, std2 = np.std(group_data[g1]), np.std(group_data[g2])
                        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                        
                        pairwise.append({
                            'group1': g1,
                            'group2': g2,
                            'test': test_pw,
                            'statistic': stat_pw,
                            'p_value': p_pw,
                            'significant': p_pw < 0.05,
                            'cohens_d': cohens_d,
                            'effect_size_interpretation': self._interpret_cohens_d(cohens_d)
                        })
                
                statistical_results['pairwise_results'].append({
                    'feature': feature,
                    'comparisons': pairwise
                })
            
            # Print summary
            sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"{feature:30s} | {test_name:15s} | p={p_value:.4f} {sig_marker}")
        
        # Create comprehensive visualization
        self._plot_cluster_distributions(
            feature_cols, 
            statistical_results['anova_results'],
            save_path
        )
        
        # Generate report
        if save_path:
            report_path = Path(save_path).parent / f"cluster_statistical_report_{model_name or 'default'}.md"
            self._generate_cluster_report(statistical_results, report_path, model_name)
        
        print(f"\n[OK] Cluster quantification analysis complete")
        print(f"    Significant features: {sum(1 for r in statistical_results['anova_results'] if r['significant'])}/{len(feature_cols)}")
        
        return statistical_results
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        d_abs = abs(d)
        if d_abs < 0.2:
            return "negligible"
        elif d_abs < 0.5:
            return "small"
        elif d_abs < 0.8:
            return "medium"
        else:
            return "large"
    
    def _plot_cluster_distributions(self, feature_cols: List[str], 
                                   anova_results: List[Dict],
                                   save_path: str = None):
        """Create boxplots showing feature distributions across clusters."""
        # Select top 12 most significant features
        sorted_results = sorted(anova_results, key=lambda x: x['p_value'])
        top_features = [r['feature'] for r in sorted_results[:12]]
        
        # Create subplot grid
        n_features = len(top_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        fig.suptitle(f'Feature Distributions Across Performance Clusters\n'
                    f'Top {n_features} Most Discriminative Features',
                    fontsize=16, fontweight='bold')
        
        for idx, feature in enumerate(top_features):
            ax = axes[idx]
            
            # Get result for this feature
            result = next(r for r in anova_results if r['feature'] == feature)
            
            # Create boxplot
            data_to_plot = []
            labels = []
            for group in ['High_Performance', 'Medium_Performance', 'Low_Performance']:
                mask = self.performance_data['performance_group'] == group
                data_to_plot.append(self.performance_data.loc[mask, feature].values)
                labels.append(group.replace('_', '\n'))
            
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                           boxprops=dict(facecolor='lightblue', alpha=0.7),
                           medianprops=dict(color='red', linewidth=2),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5))
            
            # Color by significance
            colors = ['#2ecc71', '#f39c12', '#e74c3c']  # green, orange, red
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            # Add title with test result
            sig_marker = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else "n.s."
            ax.set_title(f"{feature}\n{result['test']}: p={result['p_value']:.4f} {sig_marker}",
                        fontsize=10, fontweight='bold')
            ax.set_ylabel('Value')
            ax.grid(axis='y', alpha=0.3)
            
            # Rotate x labels
            ax.tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            boxplot_path = Path(save_path).parent / f"cluster_feature_distributions_{Path(save_path).stem}.png"
            plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Boxplot saved to {boxplot_path}")
            plt.close()
        else:
            plt.show()
    
    def _generate_cluster_report(self, results: Dict, report_path: Path, model_name: str = None):
        """Generate markdown report for cluster analysis."""
        lines = [
            f"# Cluster Quantification Statistical Report",
            f"",
            f"**Model**: {model_name or 'Default'}",
            f"**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Experiment**: {self.experiment_name}",
            f"",
            f"## Overview",
            f"",
            f"This report quantifies the differences between performance clusters using",
            f"rigorous statistical tests and effect size calculations.",
            f"",
            f"## Cluster Definitions",
            f"",
            f"- **High Performance**: Low MAE (≤ 33rd percentile)",
            f"- **Medium Performance**: Medium MAE (33rd-67th percentile)",
            f"- **Low Performance**: High MAE (≥ 67th percentile)",
            f"",
            f"## Statistical Test Results",
            f"",
            f"### Significant Features (p < 0.05)",
            f""
        ]
        
        # Add significant features
        sig_results = [r for r in results['anova_results'] if r['significant']]
        sig_results.sort(key=lambda x: x['p_value'])
        
        lines.append(f"| Feature | Test | Statistic | p-value | High Mean±SD | Medium Mean±SD | Low Mean±SD |")
        lines.append(f"|---------|------|-----------|---------|--------------|----------------|-------------|")
        
        for r in sig_results:
            lines.append(
                f"| {r['feature']} | {r['test']} | {r['statistic']:.3f} | {r['p_value']:.4f} | "
                f"{r['group_means']['High_Performance']:.2f}±{r['group_stds']['High_Performance']:.2f} | "
                f"{r['group_means']['Medium_Performance']:.2f}±{r['group_stds']['Medium_Performance']:.2f} | "
                f"{r['group_means']['Low_Performance']:.2f}±{r['group_stds']['Low_Performance']:.2f} |"
            )
        
        # Add pairwise comparisons
        lines.extend([
            f"",
            f"## Post-Hoc Pairwise Comparisons",
            f""
        ])
        
        for pw_result in results['pairwise_results']:
            lines.append(f"### {pw_result['feature']}")
            lines.append(f"")
            lines.append(f"| Comparison | Test | p-value | Cohen's d | Effect Size |")
            lines.append(f"|------------|------|---------|-----------|-------------|")
            
            for comp in pw_result['comparisons']:
                g1_short = comp['group1'].replace('_Performance', '')
                g2_short = comp['group2'].replace('_Performance', '')
                sig = "*" if comp['significant'] else ""
                lines.append(
                    f"| {g1_short} vs {g2_short} | {comp['test']} | "
                    f"{comp['p_value']:.4f}{sig} | {comp['cohens_d']:.3f} | "
                    f"{comp['effect_size_interpretation']} |"
                )
            lines.append(f"")
        
        # Save report
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"[OK] Statistical report saved to {report_path}")
    
    def fit_explanatory_model(self, 
                             model_name: str = None,
                             save_path: str = None,
                             model_type: str = 'both'):
        """
        Fit explanatory models to understand feature importance for MAE prediction.
        
        Performs:
        1. Linear regression with regularization
        2. Random forest for non-linear relationships
        3. Feature importance ranking
        4. Model performance metrics (R², RMSE, MAE)
        
        Args:
            model_name: Model to analyze
            save_path: Path to save results
            model_type: 'linear', 'tree', or 'both'
            
        Returns:
            Dictionary with model results and feature importances
        """
        print(f"\n{'='*60}")
        print("EXPLANATORY MODEL - FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*60}")
        
        if self.performance_data is None:
            self.create_performance_dataframe(model_name)
        
        # Prepare features and target
        feature_cols = [col for col in self.performance_data.columns 
                       if col not in ['patient_id', 'mae', 'rmse', 'mape', 'mard', 'tir', 
                                     'hypo_events', 'hyper_events', 'clarke_a_b', 'parkes_a_b',
                                     'n_predictions', 'performance_group'] 
                       and not col.startswith('clarke_zones_') 
                       and not col.startswith('parkes_zones_')]
        
        # Filter low-variance features
        X = self.performance_data[feature_cols].values
        feature_variances = np.var(X, axis=0)
        valid_features = feature_variances > 1e-6
        feature_cols = [feature_cols[i] for i in range(len(feature_cols)) if valid_features[i]]
        X = X[:, valid_features]
        
        y = self.performance_data['mae'].values
        
        print(f"Training explanatory models with {len(feature_cols)} features")
        print(f"Target: MAE (range: {np.min(y):.3f} - {np.max(y):.3f} mg/dL)")
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        results = {
            'feature_names': feature_cols,
            'target_stats': {
                'mean': np.mean(y),
                'std': np.std(y),
                'min': np.min(y),
                'max': np.max(y)
            }
        }
        
        # Model 1: Ridge Regression (L2 regularization)
        if model_type in ['linear', 'both']:
            from sklearn.linear_model import RidgeCV
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            
            print(f"\n--- Ridge Regression ---")
            ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=3)
            ridge.fit(X_scaled, y)
            
            y_pred = ridge.predict(X_scaled)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            
            # Feature importances (absolute coefficients)
            feature_importance = np.abs(ridge.coef_)
            feature_ranking = sorted(zip(feature_cols, feature_importance, ridge.coef_), 
                                   key=lambda x: x[1], reverse=True)
            
            results['ridge'] = {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'best_alpha': ridge.alpha_,
                'coefficients': ridge.coef_.tolist(),
                'feature_importance': feature_ranking
            }
            
            print(f"R² Score: {r2:.4f}")
            print(f"RMSE: {rmse:.4f} mg/dL")
            print(f"MAE: {mae:.4f} mg/dL")
            print(f"Best alpha: {ridge.alpha_}")
            print(f"\nTop 10 Most Important Features:")
            for i, (feat, importance, coef) in enumerate(feature_ranking[:10], 1):
                direction = "↑" if coef > 0 else "↓"
                print(f"  {i:2d}. {feat:30s} | Importance: {importance:.4f} | Coef: {coef:+.4f} {direction}")
        
        # Model 2: Random Forest
        if model_type in ['tree', 'both']:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            
            print(f"\n--- Random Forest ---")
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X, y)  # Use original scale for tree-based model
            
            y_pred_rf = rf.predict(X)
            r2_rf = r2_score(y, y_pred_rf)
            rmse_rf = np.sqrt(mean_squared_error(y, y_pred_rf))
            mae_rf = mean_absolute_error(y, y_pred_rf)
            
            # Feature importances
            feature_importance_rf = rf.feature_importances_
            feature_ranking_rf = sorted(zip(feature_cols, feature_importance_rf), 
                                       key=lambda x: x[1], reverse=True)
            
            results['random_forest'] = {
                'r2': r2_rf,
                'rmse': rmse_rf,
                'mae': mae_rf,
                'feature_importance': feature_ranking_rf
            }
            
            print(f"R² Score: {r2_rf:.4f}")
            print(f"RMSE: {rmse_rf:.4f} mg/dL")
            print(f"MAE: {mae_rf:.4f} mg/dL")
            print(f"\nTop 10 Most Important Features:")
            for i, (feat, importance) in enumerate(feature_ranking_rf[:10], 1):
                print(f"  {i:2d}. {feat:30s} | Importance: {importance:.4f}")
        
        # Create visualization
        self._plot_feature_importance(results, save_path, model_type)
        
        # Generate report
        if save_path:
            report_path = Path(save_path).parent / f"explanatory_model_report_{model_name or 'default'}.md"
            self._generate_model_report(results, report_path, model_name, model_type)
        
        print(f"\n[OK] Explanatory model analysis complete")
        
        return results
    
    def _plot_feature_importance(self, results: Dict, save_path: str = None, model_type: str = 'both'):
        """Create feature importance visualization."""
        if model_type == 'both':
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        else:
            fig, axes = plt.subplots(1, 1, figsize=(10, 8))
            axes = [axes]
        
        fig.suptitle('Feature Importance for MAE Prediction', fontsize=16, fontweight='bold')
        
        plot_idx = 0
        
        # Ridge regression plot
        if 'ridge' in results:
            ax = axes[plot_idx]
            plot_idx += 1
            
            top_features = results['ridge']['feature_importance'][:15]
            features = [f[0] for f in top_features]
            coefficients = [f[2] for f in top_features]
            
            colors = ['red' if c > 0 else 'blue' for c in coefficients]
            bars = ax.barh(range(len(features)), coefficients, color=colors, alpha=0.7)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features, fontsize=9)
            ax.set_xlabel('Coefficient (standardized)', fontsize=11)
            ax.set_title(f"Ridge Regression (R² = {results['ridge']['r2']:.3f})\n"
                        f"Red = increases MAE, Blue = decreases MAE", 
                        fontsize=12, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(axis='x', alpha=0.3)
        
        # Random forest plot
        if 'random_forest' in results:
            ax = axes[plot_idx]
            
            top_features_rf = results['random_forest']['feature_importance'][:15]
            features_rf = [f[0] for f in top_features_rf]
            importances_rf = [f[1] for f in top_features_rf]
            
            bars_rf = ax.barh(range(len(features_rf)), importances_rf, color='green', alpha=0.7)
            ax.set_yticks(range(len(features_rf)))
            ax.set_yticklabels(features_rf, fontsize=9)
            ax.set_xlabel('Feature Importance', fontsize=11)
            ax.set_title(f"Random Forest (R² = {results['random_forest']['r2']:.3f})\n"
                        f"Importance = mean decrease in impurity", 
                        fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            importance_path = Path(save_path).parent / f"feature_importance_{Path(save_path).stem}.png"
            plt.savefig(importance_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Feature importance plot saved to {importance_path}")
            plt.close()
        else:
            plt.show()
    
    def _generate_model_report(self, results: Dict, report_path: Path, 
                              model_name: str = None, model_type: str = 'both'):
        """Generate markdown report for explanatory models."""
        lines = [
            f"# Explanatory Model Report",
            f"",
            f"**Model**: {model_name or 'Default'}",
            f"**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Experiment**: {self.experiment_name}",
            f"",
            f"## Overview",
            f"",
            f"This report presents explanatory models that predict patient MAE from",
            f"glucose characteristics. The models provide quantitative evidence for",
            f"which features drive forecast difficulty.",
            f"",
            f"## Target Variable (MAE)",
            f"",
            f"- Mean: {results['target_stats']['mean']:.3f} mg/dL",
            f"- Std: {results['target_stats']['std']:.3f} mg/dL",
            f"- Range: [{results['target_stats']['min']:.3f}, {results['target_stats']['max']:.3f}] mg/dL",
            f""
        ]
        
        # Ridge regression results
        if 'ridge' in results:
            r = results['ridge']
            lines.extend([
                f"## Ridge Regression Results",
                f"",
                f"### Model Performance",
                f"",
                f"- **R² Score**: {r['r2']:.4f}",
                f"- **RMSE**: {r['rmse']:.4f} mg/dL",
                f"- **MAE**: {r['mae']:.4f} mg/dL",
                f"- **Best Alpha**: {r['best_alpha']}",
                f"",
                f"**Interpretation**: The model explains {r['r2']*100:.1f}% of the variance in patient MAE.",
                f"",
                f"### Top 15 Features (by absolute coefficient)",
                f"",
                f"| Rank | Feature | Coefficient | Abs. Importance | Effect |",
                f"|------|---------|-------------|-----------------|--------|"
            ])
            
            for i, (feat, importance, coef) in enumerate(r['feature_importance'][:15], 1):
                effect = "Increases MAE" if coef > 0 else "Decreases MAE"
                lines.append(f"| {i} | {feat} | {coef:+.4f} | {importance:.4f} | {effect} |")
            
            lines.append(f"")
        
        # Random forest results
        if 'random_forest' in results:
            r = results['random_forest']
            lines.extend([
                f"## Random Forest Results",
                f"",
                f"### Model Performance",
                f"",
                f"- **R² Score**: {r['r2']:.4f}",
                f"- **RMSE**: {r['rmse']:.4f} mg/dL",
                f"- **MAE**: {r['mae']:.4f} mg/dL",
                f"",
                f"**Interpretation**: The model explains {r['r2']*100:.1f}% of the variance in patient MAE.",
                f"",
                f"### Top 15 Features (by importance)",
                f"",
                f"| Rank | Feature | Importance |",
                f"|------|---------|------------|"
            ])
            
            for i, (feat, importance) in enumerate(r['feature_importance'][:15], 1):
                lines.append(f"| {i} | {feat} | {importance:.4f} |")
            
            lines.append(f"")
        
        # Add interpretation
        lines.extend([
            f"## Key Findings",
            f"",
            f"### Confirmed Predictors of Forecast Difficulty",
            f""
        ])
        
        # Compare top features from both models if available
        if 'ridge' in results and 'random_forest' in results:
            ridge_top = set([f[0] for f in results['ridge']['feature_importance'][:10]])
            rf_top = set([f[0] for f in results['random_forest']['feature_importance'][:10]])
            common = ridge_top.intersection(rf_top)
            
            if common:
                lines.append(f"Features important in **both** models (robust predictors):")
                lines.append(f"")
                for feat in common:
                    lines.append(f"- `{feat}`")
                lines.append(f"")
        
        # Save report
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"[OK] Explanatory model report saved to {report_path}")
    
    def analyze_patient_subgroups(self, model_name: str = None, metric: str = 'mae'):
        """
        Analyze patient subgroups based on performance.
        
        Args:
            model_name: Model to analyze
            metric: Performance metric to use for grouping
        """
        print(f"\n{'='*60}")
        print("PATIENT SUBGROUP ANALYSIS")
        print(f"{'='*60}")
        
        if self.performance_data is None:
            self.create_performance_dataframe(model_name)
        
        # Define performance groups (tertiles)
        metric_values = self.performance_data[metric].values
        low_thresh = np.percentile(metric_values, 33.33)
        high_thresh = np.percentile(metric_values, 66.67)
        
        # Create performance groups
        performance_groups = []
        for val in metric_values:
            if val <= low_thresh:
                performance_groups.append('High_Performance')  # Low error = high performance
            elif val <= high_thresh:
                performance_groups.append('Medium_Performance')
            else:
                performance_groups.append('Low_Performance')  # High error = low performance
        
        self.performance_data['performance_group'] = performance_groups
        
        # Analyze differences between groups
        feature_cols = [col for col in self.performance_data.columns 
                       if col not in ['patient_id', 'mae', 'rmse', 'mape', 'mard', 'tir', 
                                     'hypo_events', 'hyper_events', 'clarke_a_b', 'parkes_a_b',
                                     'n_predictions', 'performance_group'] and not col.startswith('clarke_zones_') and not col.startswith('parkes_zones_')]
        
        # Filter out low-variance features for better analysis
        X_analysis = self.performance_data[feature_cols].values
        feature_variances = np.var(X_analysis, axis=0)
        valid_features = feature_variances > 1e-6
        filtered_feature_cols = [feature_cols[i] for i in range(len(feature_cols)) if valid_features[i]]
        
        print(f"Analyzing {len(filtered_feature_cols)} features across performance groups...")
        
        group_analysis = {}
        for group in ['High_Performance', 'Medium_Performance', 'Low_Performance']:
            group_data = self.performance_data[self.performance_data['performance_group'] == group]
            if len(group_data) == 0:
                continue
            
            group_analysis[group] = {
                'n_patients': len(group_data),
                'patients': group_data['patient_id'].tolist(),
                f'{metric}_mean': group_data[metric].mean(),
                f'{metric}_std': group_data[metric].std(),
                'features': {}
            }
            
            for feature in filtered_feature_cols:  # Use filtered features
                group_analysis[group]['features'][feature] = {
                    'mean': group_data[feature].mean(),
                    'std': group_data[feature].std()
                }
        
        # Statistical tests between groups
        print(f"\nPerformance Group Summary ({metric.upper()}):")
        print("-" * 50)
        for group, data in group_analysis.items():
            print(f"{group}: {data['n_patients']} patients")
            print(f"  {metric.upper()}: {data[f'{metric}_mean']:.3f} ± {data[f'{metric}_std']:.3f}")
            print(f"  Patients: {data['patients']}")
            print()
        
        # Feature comparison between high and low performance groups
        if 'High_Performance' in group_analysis and 'Low_Performance' in group_analysis:
            print("Feature Differences (High vs Low Performance):")
            print("-" * 50)
            
            high_perf_data = self.performance_data[self.performance_data['performance_group'] == 'High_Performance']
            low_perf_data = self.performance_data[self.performance_data['performance_group'] == 'Low_Performance']
            
            significant_features = []
            
            for feature in filtered_feature_cols:  # Use filtered features
                high_values = high_perf_data[feature].values
                low_values = low_perf_data[feature].values
                
                if len(high_values) >= 2 and len(low_values) >= 2:
                    t_stat, p_value = stats.ttest_ind(high_values, low_values)
                    
                    effect_size = (np.mean(high_values) - np.mean(low_values)) / np.sqrt(
                        ((len(high_values) - 1) * np.var(high_values, ddof=1) + 
                         (len(low_values) - 1) * np.var(low_values, ddof=1)) / 
                        (len(high_values) + len(low_values) - 2)
                    )
                    
                    significant_features.append({
                        'feature': feature,
                        'high_mean': np.mean(high_values),
                        'low_mean': np.mean(low_values),
                        'difference': np.mean(high_values) - np.mean(low_values),
                        't_stat': t_stat,
                        'p_value': p_value,
                        'effect_size': effect_size,
                        'significant': p_value < 0.05
                    })
            
            # Sort by p-value
            significant_features.sort(key=lambda x: x['p_value'])
            
            print("Top Discriminating Features:")
            print(f"{'Feature':<20} {'High Perf':<10} {'Low Perf':<10} {'Diff':<8} {'p-value':<8} {'Significant'}")
            print("-" * 80)
            
            for sf in significant_features[:10]:
                sig_marker = "***" if sf['p_value'] < 0.001 else "**" if sf['p_value'] < 0.01 else "*" if sf['p_value'] < 0.05 else ""
                print(f"{sf['feature']:<20} {sf['high_mean']:<10.3f} {sf['low_mean']:<10.3f} "
                      f"{sf['difference']:<8.3f} {sf['p_value']:<8.4f} {sig_marker}")
        
        return group_analysis, significant_features if 'significant_features' in locals() else []
    
    def create_patient_comparison_report(self, 
                                       model_name: str = None,
                                       output_dir: str = None):
        """
        Generate a comprehensive patient comparison report.
        
        Args:
            model_name: Model to analyze
            output_dir: Directory to save the report
        """
        if output_dir is None:
            output_dir = Path.cwd() / f"patient_analysis_{self.experiment_name}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print("GENERATING PATIENT COMPARISON REPORT")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")
        
        # Load data if not already done
        if not self.patient_results:
            self.load_experiment_results()
        
        if not self.patient_features:
            self.compute_patient_features()
        
        if self.performance_data is None:
            self.create_performance_dataframe(model_name)
        
        # 1. t-SNE visualization
        feature_correlations = self.plot_patient_tsne(
            model_name=model_name,
            save_path=output_dir / f"patient_tsne_{model_name or 'default'}.png",
            use_pca_fallback=True  # Use PCA for small datasets for better proximity
        )
        
        # 1.5. Glucose values t-SNE visualization
        patient_glucose_stats = self.plot_glucose_tsne(
            model_name=model_name,
            save_path=output_dir / f"glucose_tsne_{model_name or 'default'}.png",
            use_pca_fallback=True
        )
        
        # 2. Subgroup analysis
        group_analysis, significant_features = self.analyze_patient_subgroups(
            model_name=model_name,
            metric='mae'
        )
        
        # 3. Save detailed results
        self.performance_data.to_csv(output_dir / "patient_performance_data.csv", index=False)
        
        # Save feature correlations
        corr_df = pd.DataFrame(feature_correlations)
        corr_df.to_csv(output_dir / "feature_correlations.csv", index=False)
        
        # Save significant features
        if significant_features:
            sig_df = pd.DataFrame(significant_features)
            sig_df.to_csv(output_dir / "significant_features.csv", index=False)
        
        # Save glucose statistics
        if patient_glucose_stats:
            glucose_df = pd.DataFrame.from_dict(patient_glucose_stats, orient='index')
            glucose_df.to_csv(output_dir / "patient_glucose_stats.csv", index=True)
        
        # Save group analysis
        with open(output_dir / "group_analysis.json", 'w') as f:
            json.dump(group_analysis, f, indent=2, default=str)
        
        # 4. Create summary report
        summary_lines = [
            f"# Patient Analysis Report: {self.experiment_name}",
            f"",
            f"**Model**: {model_name or 'Default'}",
            f"**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Number of Patients**: {len(self.patient_results)}",
            f"",
            f"## Performance Summary",
            f"",
        ]
        
        # Add performance statistics
        mae_values = self.performance_data['mae'].values
        summary_lines.extend([
            f"**MAE Statistics**:",
            f"- Mean: {np.mean(mae_values):.3f} mg/dL",
            f"- Std: {np.std(mae_values):.3f} mg/dL",
            f"- Min: {np.min(mae_values):.3f} mg/dL (Patient {self.performance_data.loc[self.performance_data['mae'].idxmin(), 'patient_id']})",
            f"- Max: {np.max(mae_values):.3f} mg/dL (Patient {self.performance_data.loc[self.performance_data['mae'].idxmax(), 'patient_id']})",
            f"",
            f"## Top Feature Correlations",
            f""
        ])
        
        # Add top correlations
        for fc in feature_correlations[:5]:
            summary_lines.append(f"- **{fc['feature']}**: r={fc['correlation']:.3f}, p={fc['p_value']:.4f}")
        
        summary_lines.extend([
            f"",
            f"## Performance Groups",
            f""
        ])
        
        # Add group analysis
        for group, data in group_analysis.items():
            summary_lines.extend([
                f"### {group.replace('_', ' ')}",
                f"- Patients: {data['n_patients']} ({data['patients']})",
                f"- MAE: {data['mae_mean']:.3f} ± {data['mae_std']:.3f} mg/dL",
                f""
            ])
        
        summary_lines.extend([
            f"## Key Findings",
            f"",
            f"1. **Best Performing Patient**: {self.performance_data.loc[self.performance_data['mae'].idxmin(), 'patient_id']} (MAE: {np.min(mae_values):.3f} mg/dL)",
            f"2. **Most Challenging Patient**: {self.performance_data.loc[self.performance_data['mae'].idxmax(), 'patient_id']} (MAE: {np.max(mae_values):.3f} mg/dL)",
            f"3. **Performance Range**: {np.max(mae_values) - np.min(mae_values):.3f} mg/dL difference",
            f"",
            f"## Generated Files",
            f"",
            f"- `patient_tsne_{model_name or 'default'}.png`: Patient t-SNE (performance metrics) with feature correlations",
            f"- `glucose_tsne_{model_name or 'default'}.png`: Glucose values t-SNE visualization", 
            f"- `patient_performance_data.csv`: Complete patient dataset",
            f"- `feature_correlations.csv`: Patient feature correlations with MAE",
            f"- `significant_features.csv`: Features distinguishing high/low performers",
            f"- `patient_glucose_stats.csv`: Patient glucose distribution statistics",
            f"- `group_analysis.json`: Detailed subgroup analysis",
            f""
        ])
        
        # Save summary
        with open(output_dir / "PATIENT_ANALYSIS_SUMMARY.md", 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"\n[OK] Patient analysis report saved to {output_dir}")
        print(f"  - patient_tsne_*.png")
        print(f"  - glucose_tsne_*.png")
        print(f"  - patient_performance_data.csv")
        print(f"  - feature_correlations.csv")
        print(f"  - significant_features.csv")
        print(f"  - patient_glucose_stats.csv")
        print(f"  - group_analysis.json")
        print(f"  - PATIENT_ANALYSIS_SUMMARY.md")


def analyze_patients(experiment_dir: str, 
                    experiment_name: str = None,
                    model_name: str = None,
                    output_dir: str = None):
    """
    Convenience function to analyze patients within an experiment.
    
    Args:
        experiment_dir: Path to experiment directory
        experiment_name: Optional name for the experiment
        model_name: Model to analyze
        output_dir: Directory to save results
    """
    analyzer = PatientAnalyzer(experiment_dir, experiment_name)
    analyzer.load_experiment_results()
    analyzer.create_patient_comparison_report(model_name, output_dir)
    
    return analyzer