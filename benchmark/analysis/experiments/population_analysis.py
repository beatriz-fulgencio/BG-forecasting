"""
Population-level view of a run: train vs test glucose patterns, coloured by MAE.
"""

from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from .patient_analysis import PatientAnalyzer


class PopulationAnalysis:
    """
    Population-level train vs test comparison analysis.
    """

    def __init__(self,
                 experiment_dir: str,
                 output_dir: str,
                 data_root: str = "data",
                 mode: Optional[str] = None,
                 seed: Optional[int] = None):
        """
        Initialize population analysis.

        Args:
            experiment_dir: Directory containing experiment results
            output_dir: Directory to save analysis results
            data_root: Root directory for data (default: "data")
            mode: Training mode to analyse (regular/transfer).
            seed: Analyse this seed alone instead of the cross-seed mean.
        """
        self.experiment_dir = Path(experiment_dir)
        self.output_dir = Path(output_dir)
        self.data_root = Path(data_root)
        self.mode = mode
        self.seed = seed

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train_test_comparison").mkdir(exist_ok=True)

        print("="*80)
        print("POPULATION ANALYSIS - TRAIN VS TEST COMPARISON")
        print("="*80)
        print(f"Experiment directory: {self.experiment_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Data root: {self.data_root}")
        if self.mode:
            print(f"Training mode: {self.mode}")
        if self.seed is not None:
            print(f"Seed: {self.seed}")

    def run_train_test_comparison(self, model: str = 'RNN'):
        """
        Run train vs test glucose comparison analysis.
        """
        print(f"\\n{'='*80}")
        print(f"TRAIN VS TEST GLUCOSE COMPARISON ({model})")
        print(f"{'='*80}")
        
        try:
            # Initialize analyzer
            analyzer = PatientAnalyzer(
                str(self.experiment_dir), mode=self.mode, seed=self.seed
            )
            analyzer.load_experiment_results()
            
            # Create comparison plot
            save_path = self.output_dir / "train_test_comparison" / f"train_test_glucose_comparison_{model}.png"
            
            plot_data = analyzer.plot_train_test_glucose_comparison(
                data_root=str(self.data_root),
                model_name=model,
                save_path=str(save_path),
                max_samples=3000,  # Balanced for performance
                use_pca_fallback=True
            )
            
            print(f"✓ Train/test comparison completed for {model}")
            return str(save_path)
            
        except Exception as e:
            print(f"✗ Train/test comparison failed for {model}: {e}")
            return None
    
    def create_population_report(self, results_summary: Dict, save_path: str):
        """
        Create population analysis report.
        """
        report = []
        report.append("# Population Analysis Report")
        report.append("")
        report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Experiment**: {self.experiment_dir.name}")
        report.append("")
        
        # Overview
        report.append("## Analysis Overview")
        report.append("")
        report.append("This report contains glucose history analysis and train vs test glucose pattern comparison.")
        report.append("")
        
        # Glucose history results
        if 'glucose_history' in results_summary:
            glucose_success = results_summary['glucose_history']
            report.append("## Glucose History Analysis")
            report.append("")
            if glucose_success:
                report.append("✓ **Status**: Successfully completed")
                report.append("")
                report.append("- **Population overview plot**: `glucose_history/population_glucose_overview.png`")
                report.append("- **Summary statistics**: `glucose_history/population_glucose_summary.csv`")
                report.append("- **Analysis**: Complete glucose time series for all patients showing train/test splits")
            else:
                report.append("✗ **Status**: Failed or no data available")
                report.append("")
                report.append("- Raw glucose data may not be available in the expected location")
                report.append("- Check data directory structure: `data/raw/ohiot1dm/`")
            report.append("")
        
        # Train/test comparison results
        if 'train_test_comparison' in results_summary:
            comparison_results = results_summary['train_test_comparison']
            report.append("## Train vs Test Comparison Results")
            report.append("")
            
            for model, plot_path in comparison_results.items():
                if plot_path:
                    report.append(f"### {model} Model")
                    report.append("")
                    report.append(f"- **Comparison plot**: `{Path(plot_path).name}`")
                    report.append(
                        "- **Analysis**: Two-dimensional visualization of glucose "
                        "patterns comparing training and test sets. Small cohorts may "
                        "use PCA; larger cohorts use t-SNE."
                    )
                    report.append("")
        
        # File listings
        report.append("## Generated Files")
        report.append("")
        
        # List glucose history files
        glucose_dir = self.output_dir / "glucose_history"
        if glucose_dir.exists():
            files = list(glucose_dir.rglob('*.*'))
            if files:
                report.append("### Glucose History")
                report.append("")
                for file_path in sorted(files):
                    rel_path = file_path.relative_to(self.output_dir)
                    report.append(f"- `{rel_path}`")
                report.append("")
        
        # List train/test comparison files
        subdir_path = self.output_dir / "train_test_comparison"
        if subdir_path.exists():
            files = list(subdir_path.rglob('*.*'))
            if files:
                report.append("### Train Test Comparison")
                report.append("")
                for file_path in sorted(files):
                    rel_path = file_path.relative_to(self.output_dir)
                    report.append(f"- `{rel_path}`")
                report.append("")
        
        # Write report.  ``save_path`` may be a nested path outside output_dir.
        report_path = Path(save_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open('w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"✓ Population analysis report saved: {Path(save_path).name}")
    
