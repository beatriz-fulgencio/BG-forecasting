"""
Patient-level analysis for understanding performance differences within experiments.

This module provides tools for:
- Analyzing which patients perform better/worse
- t-SNE visualization of patient characteristics
- Correlation analysis between patient features and model performance
- Statistical analysis of patient subgroups
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import List, Optional, Tuple
import json

from ..dataset.patient_features import compute_patient_features
from ..results_io import ExperimentResults, load_experiment_results


def deterministic_glucose_windows(values: np.ndarray, *, window_size: int = 12,
                                  max_windows: int = 1000) :
    """Return evenly subsampled, chronological glucose windows with no RNG.
    """
    series = np.asarray(values, dtype=float)
    if window_size < 2 or max_windows < 1:
        raise ValueError("window_size must be >= 2 and max_windows must be positive")
    if series.ndim != 1 or len(series) < window_size or not np.all(np.isfinite(series)):
        return np.empty((0, window_size), dtype=float)
    windows = np.lib.stride_tricks.sliding_window_view(series, window_size)
    if len(windows) <= max_windows:
        return windows.copy()
    indices = np.linspace(0, len(windows) - 1, num=max_windows, dtype=int)
    return windows[indices].copy()


def leave_one_patient_out_explanatory_models(
    frame: pd.DataFrame, feature_columns: List[str], *, outcome: str = "mae", random_seed: int = 42
) :
    """Return one held-out outcome prediction per patient and model.

    This is exploratory at the small cohort size. Feature rankings and errors
    are reported, but neither coefficient nor importance is a confirmatory
    finding.  Models are intentionally fitted inside each Leave-One-Out split.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import LeaveOneOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    needed = ["patient_id", outcome, *feature_columns]
    if any(column not in frame for column in needed):
        return pd.DataFrame(columns=["patient_id", "model", "observed", "predicted", "absolute_error"])
    data = frame.loc[:, needed].dropna()
    if len(data) < 3:
        return pd.DataFrame(columns=["patient_id", "model", "observed", "predicted", "absolute_error"])
    X = data[feature_columns].to_numpy(dtype=float)
    y = data[outcome].to_numpy(dtype=float)
    rows = []
    models = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, random_state=random_seed, n_jobs=1
        ),
    }
    for name, estimator in models.items():
        for train, test in LeaveOneOut().split(X):
            estimator.fit(X[train], y[train])
            predicted = float(estimator.predict(X[test])[0])
            observed = float(y[test][0])
            rows.append({
                "patient_id": int(data.iloc[test[0]]["patient_id"]), "model": name,
                "observed": observed, "predicted": predicted,
                "absolute_error": abs(predicted - observed),
                "evidence_level": "exploratory_held_out_patient",
            })
    return pd.DataFrame(rows)


class PatientAnalyzer:
    """
    Analyze patient-level performance differences within a single experiment.
    """
    
    # Metric entries that are not scalar columns
    NON_SCALAR_METRIC_KEYS = frozenset({
        "y_true", "predictions", "n_predictions", "dispersion", "seeds",
    })

    # Only these held-out glucose-signal characteristics may be used as
    # explanatory features.
    GLUCOSE_FEATURE_COLUMNS = (
        "mean_glucose", "std_glucose", "iqr_glucose", "mean_change",
        "std_change", "hypo_percent", "in_range_percent", "hyper_percent",
        "severe_hypo_percent", "severe_hyper_percent", "stability_score",
    )

    def __init__(self,
                 experiment_dir: str,
                 experiment_name: str = None,
                 mode: Optional[str] = None,
                 seed: Optional[int] = None):
        """
        Initialize the patient analyzer.

        Args:
            experiment_dir: Path to an experiment directory
            experiment_name: Optional label.
            mode: Training mode to analyse (regular/transfer).
            seed: Analyse this seed alone instead of the cross-seed mean.
        """
        self.experiment_dir = Path(experiment_dir)
        self.mode = mode
        self.seed = seed

        self.results: Optional[ExperimentResults] = None
        self.experiment_name = experiment_name
        self.patient_results = {}
        self.patient_features = {}
        self.performance_data = None

    @property
    def horizon_minutes(self):
        """Prediction horizon of the loaded run, once it has been loaded."""
        return self.results.horizon_minutes if self.results else None

    @property
    def models(self):
        """Model names present in the loaded run."""
        return self.results.models if self.results else ()

    def load_experiment_results(self):
        """Load patient metrics through the shared results reader."""
        self.results = load_experiment_results(
            self.experiment_dir, mode=self.mode, seed=self.seed
        )
        self.mode = self.results.mode
        if self.experiment_name is None:
            self.experiment_name = self.results.label

        # Analyses below mutate their rows (adding derived columns), so hand out copies rather than aliases into the loaded results.
        self.patient_results = {
            patient_id: {model: dict(metrics) for model, metrics in per_model.items()}
            for patient_id, per_model in self.results.patient_metrics.items()
        }

        seed_note = (
            f"seed {self.results.seeds[0]}"
            if self.results.aggregation == "single_seed" and self.results.seeds
            else f"mean of {len(self.results.seeds)} seed(s)"
            if self.results.aggregation == "seed_mean"
            else "single run"
        )
        print(
            f"[OK] Loaded {len(self.patient_results)} patients from {self.experiment_name} "
            f"({', '.join(self.results.models)}; {seed_note})"
        )
        return self.patient_results

    def compute_patient_features(self):
        """Describe each patient's held-out glucose series."""
        self.patient_features = compute_patient_features(self.results)

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
                # Raw series, per-seed spread and seed lists are carried on the metrics for other callers
                if key in self.NON_SCALAR_METRIC_KEYS:
                    continue
                if isinstance(value, dict):
                    # Flatten nested dictionaries (tir, clarke_zones, parkes_zones)
                    for sub_key, sub_value in value.items():
                        row[f"{key}_{sub_key}"] = sub_value
                else:
                    row[key] = value
            
            data.append(row)
        
        self.performance_data = pd.DataFrame(data)
        print(f"[DONE] Created performance DataFrame with {len(self.performance_data)} patients and {len(self.performance_data.columns)} features")
        
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
        
        # Embedding and clustering are based only on lucose signal characteristics
        feature_cols = [
            column for column in self.GLUCOSE_FEATURE_COLUMNS if column in self.performance_data
        ]
        
        # Prepare data
        X = self.performance_data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
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
        
        print(f"Running dimensionality reduction on {len(patient_ids)} patients with {len(feature_cols)} glucose-derived features...")
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
        fig.suptitle(f'Patient Glucose-Feature Analysis (exploratory): {self.experiment_name}\nModel: {model_name or "Default"} - {method_name} {explained_variance}',
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Colored by MAE performance
        metric_values = self.performance_data['mae'].values
        
        scatter1 = axes[0, 0].scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                     c=metric_values, cmap='viridis_r', 
                                     s=100, alpha=0.7, edgecolors='black', linewidth=1)
        axes[0, 0].set_title('Patients in Glucose-Feature Space\n(Colored by MAE)',
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
        
        # Plot 4: Correlate held-out glucose characteristics with MAE. Do not
        # include model-output metrics such as TIR or error-grid zones: those
        # are alternative measures of forecasting performance, not patient
        # predictors.
        all_feature_cols = [
            column for column in self.GLUCOSE_FEATURE_COLUMNS if column in self.performance_data
        ]
        
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
    
    def _plot_glucose_tsne_legacy(self,
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
        rng = np.random.default_rng(42)
        
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
                indices = rng.choice(len(y_true), max_samples, replace=False)
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
            pca = PCA(n_components=2, random_state=42)
            glucose_reduced = pca.fit_transform(glucose_scaled)
            method_name = "PCA"
            explained_variance = f"(Explained variance: {pca.explained_variance_ratio_[0]:.2f}, {pca.explained_variance_ratio_[1]:.2f})"
        else:
            # Use t-SNE with appropriate perplexity
            optimal_perplexity = min(max(5, len(glucose_data) // 100), 50)
            print(f"Using t-SNE with perplexity: {optimal_perplexity}")
            
            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=optimal_perplexity,
                max_iter=1000,
                learning_rate='auto',
                init='pca',
                metric='euclidean'
            )
            glucose_reduced = tsne.fit_transform(glucose_scaled)
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
    
    
    def plot_glucose_tsne(self,
                          model_name: str = None,
                          save_path: str = None,
                          max_samples: int = 1000,
                          use_pca_fallback: bool = True,
                          random_state: int = 42):
        """Embed test-period glucose values per patient, coloured by MAE. """
        print(f"\n{'='*60}")
        print("GLUCOSE VALUES t-SNE ANALYSIS")
        print(f"{'='*60}")

        if not self.patient_results:
            self.load_experiment_results()

        rng = np.random.default_rng(random_state)

        glucose_values, patient_labels, mae_values = [], [], []
        # The figure is titled with the model whose MAE colours it.
        model_label = model_name or ""
        for patient_id in sorted(self.patient_results.keys()):
            models = self.patient_results[patient_id]
            if model_name is None:
                target_model = next(iter(models))
            elif model_name in models:
                target_model = model_name
            else:
                continue
            model_label = model_label or target_model

            y_true = models[target_model].get("y_true")
            mae = models[target_model].get("mae")
            if y_true is None or not len(y_true) or mae is None:
                print(f"    Warning: no glucose data available for patient {patient_id}")
                continue

            y_true = np.asarray(y_true, dtype=float)
            if len(y_true) > max_samples:
                # Sorted so the retained readings stay in chronological order:
                # the embedding's second coordinate is positional, and shuffling
                # here would scramble it into noise.
                indices = np.sort(rng.choice(len(y_true), max_samples, replace=False))
                sampled = y_true[indices]
            else:
                sampled = y_true

            glucose_values.append(sampled)
            patient_labels.extend([patient_id] * len(sampled))
            mae_values.extend([float(mae)] * len(sampled))

        if not glucose_values:
            print("    Error: no glucose data available for analysis")
            return None

        glucose_array = np.concatenate(glucose_values).reshape(-1, 1)
        patient_array = np.asarray(patient_labels)
        mae_array = np.asarray(mae_values, dtype=float)
        n_patients = len(np.unique(patient_array))
        print(f"Analyzing {len(glucose_array)} glucose values from {n_patients} patients...")

        glucose_scaled = StandardScaler().fit_transform(glucose_array).flatten()

        if use_pca_fallback and n_patients < 8:
            # A cohort this small gives t-SNE too few neighbours to be meaningful.
            print("Using PCA instead of t-SNE for small dataset")
            from sklearn.decomposition import PCA
            features = np.column_stack([
                glucose_scaled, rng.normal(0, 0.1, len(glucose_scaled)),
            ])
            reducer = PCA(n_components=2, random_state=random_state)
            embedding = reducer.fit_transform(features)
            method = "PCA"
            variance = ("(explained variance: "
                        f"{reducer.explained_variance_ratio_[0]:.2f}, "
                        f"{reducer.explained_variance_ratio_[1]:.2f})")
        else:
            perplexity = min(max(5, len(glucose_scaled) // 100), 50)
            print(f"Using t-SNE with perplexity: {perplexity}")
            position = np.arange(len(glucose_scaled)) / len(glucose_scaled)
            features = np.column_stack([glucose_scaled, position])
            embedding = TSNE(
                n_components=2, random_state=random_state, perplexity=perplexity,
                max_iter=1000, learning_rate="auto", init="pca", metric="euclidean",
            ).fit_transform(features)
            method = "t-SNE"
            variance = ""

        figure, axes = plt.subplots(1, 2, figsize=(16, 6))
        horizon = f", {self.horizon_minutes}-min" if self.horizon_minutes else ""
        figure.suptitle(
            f"Test-period glucose values ({model_label}{horizon}): {method} embedding "
            f"(exploratory) {variance}\n{self.experiment_name}".rstrip(),
            fontsize=13, fontweight="bold",
        )

        # Left: coloured by patient. Legend carries each patient's MAE so the two
        # panels can be read against each other without matching colours by eye.
        palette = plt.cm.tab20(np.linspace(0, 1, max(n_patients, 1)))
        for index, patient_id in enumerate(sorted(np.unique(patient_array))):
            mask = patient_array == patient_id
            axes[0].scatter(embedding[mask, 0], embedding[mask, 1],
                            color=[palette[index]], s=20, alpha=.6, edgecolors="none",
                            label=f"P{patient_id} (MAE {mae_array[mask][0]:.1f})")
        axes[0].set_title("Glucose values coloured by patient", fontsize=12, fontweight="bold")
        if n_patients <= 15:
            axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7,
                           framealpha=.9, title="Patient (MAE, mg/dL)")
        else:
            axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6,
                           ncol=2, framealpha=.9, title="Patient (MAE, mg/dL)")

        # Right: the same points coloured by that patient's MAE. MAE is applied
        # after the embedding is fitted; it is never a coordinate.
        colored = axes[1].scatter(embedding[:, 0], embedding[:, 1], c=mae_array,
                                  cmap="viridis_r", s=20, alpha=.6, edgecolors="none")
        axes[1].set_title("Glucose values coloured by patient MAE (not an input)",
                          fontsize=12, fontweight="bold")
        figure.colorbar(colored, ax=axes[1], label="MAE (mg/dL)")

        for axis in axes:
            axis.set_xlabel(f"{method} component 1")
            axis.set_ylabel(f"{method} component 2")
        figure.tight_layout()
        if save_path:
            figure.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(figure)
            print(f"[OK] Glucose t-SNE plot saved to {save_path}")
        else:
            plt.close(figure)

        # Per-patient glucose distribution of the embedded readings, alongside the
        # MAE that colours them: enough to check from the returned summary whether
        # a cluster is a level effect or a variability one, without reopening the
        # figure.
        print("\nPatient glucose distribution:")
        print("-" * 50)
        summary = {}
        for patient_id in np.unique(patient_array):
            mask = patient_array == patient_id
            readings = glucose_array[mask].flatten()
            summary[int(patient_id)] = {
                "n_samples": int(readings.size),
                "mae": float(mae_array[mask][0]),
                "mean_glucose": float(np.mean(readings)),
                "std_glucose": float(np.std(readings)),
                "min_glucose": float(np.min(readings)),
                "max_glucose": float(np.max(readings)),
            }
            entry = summary[int(patient_id)]
            print(f"Patient {patient_id}: mean={entry['mean_glucose']:.1f}, "
                  f"std={entry['std_glucose']:.1f}, "
                  f"range=[{entry['min_glucose']:.0f}-{entry['max_glucose']:.0f}], "
                  f"MAE={entry['mae']:.3f}, n={entry['n_samples']}")
        print("[OK] Glucose t-SNE analysis complete")
        return summary

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
        
        """
        from scipy.spatial import ConvexHull
        from matplotlib.patches import Polygon
        
        print(f"\n{'='*60}")
        print(f"Creating Cluster-t-SNE Correlation Analysis")
        print(f"{'='*60}")
        
        # This draws whatever create_performance_dataframe already built; it does
        # not load anything itself.
        if self.performance_data is None:
            print("[ERROR] No performance data. Call load_experiment_results() and "
                  "create_performance_dataframe() first.")
            return None

        if model_name is None:
            if not self.models:
                print("[ERROR] No model in the loaded run; pass model_name explicitly.")
                return None
            model_name = self.models[0]
        
        # Use MAE as primary metric
        metric = 'mae'
        valid_mask = self.performance_data[metric].notna()
        patient_ids = self.performance_data.loc[valid_mask, 'patient_id'].values
        metric_values = self.performance_data.loc[valid_mask, metric].values
        
        if len(patient_ids) < 3:
            print(f"[WARNING] Need at least 3 patients, found {len(patient_ids)}")
            return None
        
        # Cluster only glucose-derived characteristics.  Outcomes remain a
        # descriptive colour/summary after data-derived groups are formed.
        feature_cols = [column for column in (
            'mean_glucose', 'std_glucose', 'iqr_glucose', 'mean_change',
            'std_change', 'hypo_percent', 'in_range_percent', 'hyper_percent',
            'severe_hypo_percent', 'severe_hyper_percent', 'stability_score',
        ) if column in self.performance_data]
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
        
        if cluster_method != 'kmeans':
            raise ValueError("Only glucose-feature KMeans clustering is supported; performance-based groups are circular")
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(f'Exploratory glucose-feature clusters: {model_name}\n'
                     f'{n_clusters} data-derived clusters ({method_name})',
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
            elif ci['mean_mae'] < np.percentile(metric_values, 67):
                perf_label = "MEDIUM PERFORMANCE"
            else:
                perf_label = "LOW PERFORMANCE"

            # No emoji here: this string is rendered into the figure by matplotlib,
            # whose default mono font has no glyph for them, so they came out as
            # blank boxes and warned on every call. The label carries the meaning.
            mapping_text += f"Cluster {ci['cluster']} - {perf_label}\n"
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
            # Was os.makedirs/os.path.join, but this module never imported os, so
            # the default-path branch raised NameError on every call. Uses the
            # pathlib import the rest of the module already relies on.
            directory = Path(output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            save_path = str(
                directory / f'cluster_tsne_correlation_{model_name}_{method.lower()}.png'
            )
        
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
                
                # Each observation is a deterministic 12-reading history, not
                # an individual value paired with an invented row-order clock.
                windows = deterministic_glucose_windows(
                    glucose_values, window_size=12, max_windows=max_samples
                )
                if not len(windows):
                    continue
                glucose_data.extend(windows)
                patient_labels.extend([patient_id] * len(windows))
                mae_values.extend([patient_mae[patient_id]] * len(windows))
            
            plot_data[dataset_type] = {
                'glucose': np.array(glucose_data),
                'patients': np.array(patient_labels),
                'mae': np.array(mae_values)
            }
            
            print(f"{dataset_type.capitalize()}: {len(glucose_data)} glucose windows from {len(set(patient_labels))} patients")
        
        # Create t-SNE/PCA embeddings for both datasets
        n_patients = len(patient_ids)
        use_pca = use_pca_fallback and n_patients < 8
        method_name = "PCA" if use_pca else "t-SNE"
        
        embeddings = {}
        for dataset_type in ['train', 'test']:
            glucose_array = plot_data[dataset_type]['glucose']
            
            # Standardize
            scaler = StandardScaler()
            glucose_scaled = scaler.fit_transform(glucose_array)
            
            if use_pca:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2, random_state=42)
                embedding = pca.fit_transform(glucose_scaled)
                explained_var = f"(Var: {pca.explained_variance_ratio_[0]:.2f}, {pca.explained_variance_ratio_[1]:.2f})"
            else:
                optimal_perplexity = min(50, max(2, (len(glucose_array) - 1) // 3))
                
                tsne = TSNE(
                    n_components=2,
                    random_state=42,
                    perplexity=optimal_perplexity,
                    max_iter=1000,
                    learning_rate='auto',
                    init='pca',
                    metric='euclidean'
                )
                embedding = tsne.fit_transform(glucose_scaled)
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
        
        # Restrict explanatory analysis to held-out glucose characteristics.
        # Including model-output metrics here would compare MAE against another
        # measurement of prediction quality rather than a patient feature.
        feature_cols = [
            column for column in self.GLUCOSE_FEATURE_COLUMNS if column in self.performance_data
        ]
        
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
                high_values = pd.to_numeric(high_perf_data[feature], errors='coerce').to_numpy()
                low_values = pd.to_numeric(low_perf_data[feature], errors='coerce').to_numpy()
                high_values = high_values[np.isfinite(high_values)]
                low_values = low_values[np.isfinite(low_values)]
                
                if len(high_values) >= 2 and len(low_values) >= 2:
                    t_stat, p_value = stats.ttest_ind(
                        high_values, low_values, equal_var=False
                    )

                    pooled_std = np.sqrt(
                        ((len(high_values) - 1) * np.var(high_values, ddof=1) + 
                         (len(low_values) - 1) * np.var(low_values, ddof=1)) / 
                         (len(high_values) + len(low_values) - 2)
                    )
                    difference = float(np.mean(high_values) - np.mean(low_values))
                    effect_size = difference / pooled_std if pooled_std > 0 else np.nan
                    
                    significant_features.append({
                        'feature': feature,
                        'high_mean': np.mean(high_values),
                        'low_mean': np.mean(low_values),
                        'difference': difference,
                        't_stat': t_stat,
                        'p_value': p_value,
                        'effect_size': effect_size,
                        'significant_raw': p_value < 0.05,
                        'grouping_note': 'exploratory; groups were derived from MAE',
                    })

            # These are data-derived subgroups and many feature screens, so
            # they are exploratory even after correction.  BH controls the
            # named within-report feature family; raw p-values remain visible.
            if significant_features:
                from statsmodels.stats.multitest import multipletests
                raw = np.asarray([item['p_value'] for item in significant_features], dtype=float)
                adjusted = np.full(raw.shape, np.nan, dtype=float)
                valid = np.isfinite(raw)
                if valid.any():
                    adjusted[valid] = multipletests(raw[valid], method='fdr_bh')[1]
                for item, value in zip(significant_features, adjusted):
                    item['p_value_bh'] = float(value)
                    item['significant_bh'] = bool(np.isfinite(value) and value < 0.05)
            
            # Sort by p-value
            significant_features.sort(key=lambda x: x['p_value'])
            
            print("Top Discriminating Features:")
            print(f"{'Feature':<20} {'High Perf':<10} {'Low Perf':<10} {'Diff':<8} {'p-value':<8} {'Significant'}")
            print("-" * 80)
            
            for sf in significant_features[:10]:
                sig_marker = "*" if sf.get('significant_bh') else ""
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
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
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

        glucose_feature_columns = [
            column for column in self.GLUCOSE_FEATURE_COLUMNS if column in self.performance_data
        ]
        loo = leave_one_patient_out_explanatory_models(
            self.performance_data, glucose_feature_columns, outcome="mae"
        )
        loo.to_csv(output_dir / "loo_explanatory_predictions.csv", index=False)
        
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
            f"- `loo_explanatory_predictions.csv`: exploratory leave-one-patient-out Ridge and forest predictions",
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
                    output_dir: str = None,
                    mode: Optional[str] = None,
                    seed: Optional[int] = None):
    """
    Convenience function to analyze patients within an experiment.

    Args:
        experiment_dir: Path to experiment directory
        experiment_name: Optional name for the experiment
        model_name: Model to analyze. Defaults to the run's only model.
        output_dir: Directory to save results
        mode: Training mode to analyse; required only for multi-mode runs
        seed: Analyse this seed alone instead of the cross-seed mean
    """
    analyzer = PatientAnalyzer(experiment_dir, experiment_name, mode=mode, seed=seed)
    analyzer.load_experiment_results()
    analyzer.create_patient_comparison_report(model_name, output_dir)
    
    return analyzer
