"""Shared configuration checks for grid training, resume, and merging."""
import argparse
from pathlib import Path
from benchmark.configs import load_config


def scientific_settings(path, seed):
    cfg = load_config(path).to_dict()
    cfg['data'].pop('root', None)
    cfg['training']['seeds'] = [seed]
    return {key: cfg[key] for key in ('data', 'preprocessing', 'model', 'training', 'evaluation')}


def matches(saved, expected, seed):
    try:
        cfg = load_config(saved).to_dict()
        return (cfg['training']['seeds'] == [seed] and
                scientific_settings(saved, seed) == scientific_settings(expected, seed))
    except (ValueError, TypeError, KeyError, OSError):
        return False


def validate(path):
    cfg = load_config(path).to_dict()
    pre, tr = cfg['preprocessing'], cfg['training']
    required = dict(regular_schedule='single_stage', epochs=200,
                    early_stopping_patience=5, transfer_early_stopping_patience=5,
                    pretrain_epochs=10, finetune_epochs=10, batch_size=16,
                    learning_rate=.0003, finetune_learning_rate=.00005,
                    weight_decay=.00001, grad_clip_norm=1.0, device='cuda', mode='both')
    for key, value in required.items():
        if tr[key] != value:
            raise ValueError(f'{path}: training.{key} must be {value!r}, got {tr[key]!r}')
    assert pre['include_feature_engineering'] and not pre['unimodal'], f'{path}: expected six features'
    assert pre['window_size'] == {3:12,6:12,9:18,12:24}[pre['prediction_horizon']]
    assert pre['sampling_rate'] == 5 and pre['normalization'] == 'standardize'
    arch = cfg['model']['architecture']
    assert all(arch[k] == v for k,v in dict(hidden_size=128,num_layers=2,dropout=.2,batch_first=True).items())
    assert cfg['data']['version'] == 'both' and cfg['data']['patients'] == 'all'
    assert cfg['data']['train_ratio'] == .9 and cfg['data']['validation_ratio'] == .1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['validate', 'completed'])
    parser.add_argument('config', type=Path)
    parser.add_argument('--results-dir', type=Path)
    parser.add_argument('--name')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.action == 'validate':
        validate(args.config)
    else:
        found = False
        for saved in args.results_dir.glob('*/resolved_config.yaml'):
            if not (saved.parent/'aggregate_metrics.json').is_file():
                continue
            if matches(saved, args.config, args.seed) and load_config(saved).experiment.name == args.name:
                found = True
                break
        raise SystemExit(0 if found else 1)
