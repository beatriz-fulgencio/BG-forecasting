"""
Command-line interface for the blood glucose forecasting benchmark.

This module provides CLI functionality for:
- Running benchmarking experiments
- Comparing model performance
- Managing configurations
- Processing datasets
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from benchmark.configs.config_manager import ConfigManager
from benchmark.experiments.runner import ExperimentRunner
from benchmark.experiments.tracking import ExperimentTracker, compare_experiments
from benchmark.data.loaders import OhioT1DMDataLoader
from benchmark.evaluation.evaluator import BGEvaluator


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Blood Glucose Forecasting Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run default experiment
  python -m benchmark.cli run

  # Run with custom configuration
  python -m benchmark.cli run --config configs/my_experiment.yaml

  # Run specific models on specific patients
  python -m benchmark.cli run --models rnn lstm --patients 540 544

  # Compare experiments
  python -m benchmark.cli compare results/experiment_20240101_120000 results/experiment_20240101_130000

  # List available datasets
  python -m benchmark.cli data list

  # Process raw data
  python -m benchmark.cli data process --patients 540 544
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run benchmarking experiments')
    run_parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file'
    )
    run_parser.add_argument(
        '--models', '-m',
        nargs='+',
        choices=['rnn', 'lstm', 'gru'],
        help='Models to train and evaluate'
    )
    run_parser.add_argument(
        '--patients', '-p',
        nargs='+',
        type=int,
        help='Patient IDs to include in experiment'
    )
    run_parser.add_argument(
        '--epochs', '-e',
        type=int,
        help='Number of training epochs'
    )
    run_parser.add_argument(
        '--batch-size', '-b',
        type=int,
        help='Training batch size'
    )
    run_parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        help='Learning rate for training'
    )
    run_parser.add_argument(
        '--sequence-length', '-sl',
        type=int,
        help='Input sequence length'
    )
    run_parser.add_argument(
        '--prediction-horizon', '-ph',
        type=int,
        help='Prediction horizon'
    )
    run_parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device to use for training'
    )
    run_parser.add_argument(
        '--no-transfer-learning',
        action='store_true',
        help='Disable transfer learning'
    )
    run_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='results',
        help='Output directory for results'
    )
    run_parser.add_argument(
        '--name',
        type=str,
        help='Experiment name'
    )
    run_parser.add_argument(
        '--tags',
        nargs='+',
        help='Tags for the experiment'
    )
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare experiment results')
    compare_parser.add_argument(
        'experiment_dirs',
        nargs='+',
        help='Paths to experiment directories to compare'
    )
    compare_parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for comparison results'
    )
    compare_parser.add_argument(
        '--format',
        choices=['json', 'csv', 'markdown'],
        default='markdown',
        help='Output format for comparison'
    )
    
    # Data command
    data_parser = subparsers.add_parser('data', help='Data management commands')
    data_subparsers = data_parser.add_subparsers(dest='data_command', help='Data operations')
    
    # Data list command
    list_parser = data_subparsers.add_parser('list', help='List available datasets')
    list_parser.add_argument(
        '--dataset',
        choices=['ohiot1dm'],
        default='ohiot1dm',
        help='Dataset to list'
    )
    
    # Data process command
    process_parser = data_subparsers.add_parser('process', help='Process raw data')
    process_parser.add_argument(
        '--patients',
        nargs='+',
        type=int,
        help='Patient IDs to process'
    )
    process_parser.add_argument(
        '--dataset',
        choices=['ohiot1dm'],
        default='ohiot1dm',
        help='Dataset to process'
    )
    process_parser.add_argument(
        '--version',
        choices=['2018', '2020'],
        default='2018',
        help='Dataset version'
    )
    process_parser.add_argument(
        '--output-dir',
        type=str,
        default='processed_data',
        help='Output directory for processed data'
    )
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_subparsers = config_parser.add_subparsers(dest='config_command', help='Config operations')
    
    # Config create command
    create_parser = config_subparsers.add_parser('create', help='Create configuration template')
    create_parser.add_argument(
        'output_path',
        help='Output path for configuration file'
    )
    create_parser.add_argument(
        '--template',
        choices=['default', 'minimal', 'transfer_learning'],
        default='default',
        help='Configuration template to use'
    )
    
    # Config validate command
    validate_parser = config_subparsers.add_parser('validate', help='Validate configuration')
    validate_parser.add_argument(
        'config_path',
        help='Path to configuration file to validate'
    )
    
    return parser


def run_experiment(args) -> int:
    """Run benchmarking experiment."""
    try:
        # Load configuration
        config_manager = ConfigManager(args.config)
        
        # Update configuration from command-line arguments
        cli_args = {
            'models': args.models,
            'patient_ids': args.patients,
            'epochs': args.epochs,
            ''
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'sequence_length': args.sequence_length,
            'prediction_horizon': args.prediction_horizon,
            'device': args.device,
            'fine_tune_epochs': (args.epochs // 2) if args.epochs and not args.no_transfer_learning else None,
        }
        
        # Remove None values
        cli_args = {k: v for k, v in cli_args.items() if v is not None}
        config_manager.update_from_args(cli_args)
        
        # Update experiment configuration
        if args.name:
            config_manager.set('experiment.name', args.name)
        if args.tags:
            config_manager.set('experiment.tags', args.tags)
        if args.no_transfer_learning:
            config_manager.set('training.transfer_learning.enabled', False)
        
        # Update output directory
        config_manager.set('paths.results', args.output_dir)
        
        # Create and run experiment
        experiment_runner = ExperimentRunner(config_manager.to_dict())
        results = experiment_runner.run_experiment()
        
        print(f"\n✅ Experiment completed successfully!")
        print(f"📊 Results summary:")
        
        # Display key results
        for model_name, model_results in results.items():
            if isinstance(model_results, dict) and 'evaluation' in model_results:
                eval_results = model_results['evaluation']
                mae = eval_results.get('mae', 'N/A')
                mard = eval_results.get('mard', 'N/A')
                print(f"  {model_name}: MAE={mae:.3f} mg/dL, MARD={mard:.2f}%" 
                      if isinstance(mae, (int, float)) and isinstance(mard, (int, float))
                      else f"  {model_name}: MAE={mae}, MARD={mard}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        return 1


def compare_experiment_results(args) -> int:
    """Compare experiment results."""
    try:
        comparison = compare_experiments(args.experiment_dirs)
        
        if 'error' in comparison:
            print(f"❌ Comparison failed: {comparison['error']}")
            return 1
        
        print(f"📊 Comparing {comparison['experiments']} experiments:")
        print(f"Experiment IDs: {', '.join(comparison['experiment_ids'])}")
        print()
        
        # Display model comparison
        if comparison['model_results']:
            print("Model Performance Comparison:")
            print("-" * 80)
            print(f"{'Model':<30} {'Experiment':<20} {'MAE':<10} {'MARD':<10} {'TIR':<10}")
            print("-" * 80)
            
            for result in comparison['model_results']:
                exp_id = result['experiment_id'][:12] + "..." if len(result['experiment_id']) > 15 else result['experiment_id']
                mae = f"{result['mae']:.3f}" if result['mae'] is not None else "N/A"
                mard = f"{result['mard']:.2f}%" if result['mard'] is not None else "N/A"
                tir = f"{result['tir']:.1f}%" if result['tir'] is not None else "N/A"
                
                print(f"{result['model']:<30} {exp_id:<20} {mae:<10} {mard:<10} {tir:<10}")
        
        # Save comparison if requested
        if args.output:
            output_path = Path(args.output)
            
            if args.format == 'json':
                import json
                with open(output_path, 'w') as f:
                    json.dump(comparison, f, indent=2, default=str)
            elif args.format == 'csv':
                import pandas as pd
                df = pd.DataFrame(comparison['model_results'])
                df.to_csv(output_path, index=False)
            elif args.format == 'markdown':
                # Create markdown report
                lines = [
                    f"# Experiment Comparison Report",
                    f"",
                    f"**Number of experiments**: {comparison['experiments']}",
                    f"**Experiment IDs**: {', '.join(comparison['experiment_ids'])}",
                    f"",
                    f"## Model Performance",
                    f"",
                    f"| Model | Experiment | MAE (mg/dL) | MARD (%) | TIR (%) |",
                    f"|-------|------------|-------------|----------|---------|"
                ]
                
                for result in comparison['model_results']:
                    mae = f"{result['mae']:.3f}" if result['mae'] is not None else "N/A"
                    mard = f"{result['mard']:.2f}" if result['mard'] is not None else "N/A"
                    tir = f"{result['tir']:.1f}" if result['tir'] is not None else "N/A"
                    
                    lines.append(f"| {result['model']} | {result['experiment_id']} | {mae} | {mard} | {tir} |")
                
                with open(output_path, 'w') as f:
                    f.write('\n'.join(lines))
            
            print(f"\n💾 Comparison saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return 1


def manage_data(args) -> int:
    """Manage data operations."""
    try:
        if args.data_command == 'list':
            # List available datasets
            print(f"📊 Available {args.dataset} data:")

            data_loader = OhioT1DMDataLoader(data_dir="data")
            available_patients = data_loader.get_available_patients('train')
            
            print(f"Available patients: {available_patients}")
            
        elif args.data_command == 'process':
            # Process raw data
            print(f"🔄 Processing {args.dataset} data...")
            
            data_loader = OhioT1DMDataLoader()
            patient_ids = args.patients or [540, 544, 552, 567, 584, 596]
            
            for patient_id in patient_ids:
                print(f"Processing patient {patient_id}...")
                
                # Load and process data
                train_data, test_data = data_loader.load_patient_data(
                    patient_id=patient_id,
                    dataset=args.dataset,
                    version=args.version
                )
                
                # Save processed data
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                train_file = output_dir / f"patient_{patient_id}_train_processed.csv"
                test_file = output_dir / f"patient_{patient_id}_test_processed.csv"
                
                train_data.to_csv(train_file, index=False)
                test_data.to_csv(test_file, index=False)
                
                print(f"  ✅ Saved: {train_file}, {test_file}")
            
            print(f"✅ Data processing completed!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Data operation failed: {e}")
        return 1


def manage_config(args) -> int:
    """Manage configuration operations."""
    try:
        if args.config_command == 'create':
            # Create configuration template
            config_manager = ConfigManager()
            
            # Modify config based on template
            if args.template == 'minimal':
                config = {
                    'data': config_manager.get_data_config(),
                    'models': {'gru': config_manager.get_model_config('gru')},
                    'training': config_manager.get_training_config(),
                    'evaluation': config_manager.get_evaluation_config(),
                    'paths': config_manager.get_paths_config()
                }
            elif args.template == 'transfer_learning':
                config = config_manager.to_dict()
                config['training']['transfer_learning']['enabled'] = True
            else:
                config = config_manager.to_dict()
            
            # Save configuration
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            import yaml
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            print(f"✅ Configuration template created: {output_path}")
            
        elif args.config_command == 'validate':
            # Validate configuration
            print(f"🔍 Validating configuration: {args.config_path}")
            
            config_manager = ConfigManager(args.config_path)
            print(f"✅ Configuration is valid!")
            
            # Show configuration summary
            print(f"\nConfiguration Summary:")
            print(f"- Dataset: {config_manager.get('data.dataset')}")
            print(f"- Patients: {config_manager.get('data.patient_ids')}")
            print(f"- Models: {list(config_manager.get('models', {}).keys())}")
            print(f"- Transfer Learning: {config_manager.get('training.transfer_learning.enabled')}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Configuration operation failed: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate handler
    if args.command == 'run':
        return run_experiment(args)
    elif args.command == 'compare':
        return compare_experiment_results(args)
    elif args.command == 'data':
        return manage_data(args)
    elif args.command == 'config':
        return manage_config(args)
    else:
        print(f"❌ Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())