#!/usr/bin/env python3
"""History-budget transfer experiment. Isolated outputs; no manuscript edits."""
from pathlib import Path
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

COHORT = {'2018': [559, 563, 570, 575, 588, 591],
          '2020': [540, 544, 552, 567, 584, 596]}
FEATURES = ['glucose', 'basal', 'bolus', 'carbs']
DEFAULTS = dict(model='gru', horizon_steps=6, window=12, sampling_minutes=5,
    hidden_size=64, layers=2, dropout=.2, batch_size=64,
    source_epochs=10, target_epochs=50, source_patience=5, target_patience=10,
    learning_rate=.001, validation_fraction=.2, budgets_days=[3, 5, 7, 10, 'full'],
    seeds=[41, 42, 43], patients=sorted(sum(COHORT.values(), [])),
    min_train_windows=32, min_validation_windows=8, min_test_windows=32,
    bootstrap_replicates=20000, analysis_seed=20260923, smoke=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load_support(folder):
    modules = []
    for name in ['loaders', 'preprocessors']:
        spec = importlib.util.spec_from_file_location('cold_' + name, Path(folder) / f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def clean_frame(raw, preprocessor):
    # Called ONCE per record, BEFORE slicing, exactly as the published pipeline
    # preprocesses a whole train/test file. Cleaning each slice separately
    # instead looks stricter but is not: basic_preprocessing forward-fills basal
    # and then drops any row still holding a NaN, so a slice containing no basal
    # event loses every row. Basal events are sparse -- a 4.8 h validation tail
    # usually holds none -- which emptied 10 of 12 patients at the 1-day budget.
    # What crosses a budget boundary here is a pump setting carried forward in
    # time, never a glucose observation: glucose is never filled (strategy
    # 'none'), and no fill moves backwards.
    with contextlib.redirect_stdout(io.StringIO()):
        df = preprocessor.preprocess_ohiot1dm_data(
            {0: raw.copy()}, include_feature_engineering=False)[0]
    if df.empty:
        raise ValueError('Empty preprocessed segment')
    # A channel is absent when the patient logged no such event at all: patient
    # 567's test record has no meals, so the loader emits no carbs column. The
    # published pipeline substitutes zeros for an absent feature column
    # (benchmark/data/torch_dataset.py), and the line below already zero-fills
    # missing bolus/carbs values, so an absent event channel is zero here too.
    # Glucose and basal are not event channels -- absence means a broken record.
    for channel in ['bolus', 'carbs']:
        if channel not in df.columns:
            df[channel] = 0.
    absent = [c for c in FEATURES if c not in df.columns]
    if absent:
        raise ValueError('Missing required channel(s): ' + ', '.join(absent))
    df = df[FEATURES].apply(pd.to_numeric, errors='coerce')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if df.index.has_duplicates:
        raise ValueError('Duplicate preprocessed timestamps')
    df = df.replace(-1, np.nan)
    df[['bolus', 'carbs']] = df[['bolus', 'carbs']].fillna(0.)
    df.loc[~df.glucose.between(20, 600), 'glucose'] = np.nan
    return df


def cgm_bounds(raw):
    glucose = pd.to_numeric(raw.glucose, errors='coerce')
    times = pd.DatetimeIndex(raw.index)[glucose.between(20, 600)]
    if len(times) < 2:
        raise ValueError('Insufficient glucose history')
    return times.min(), times.max()


def budget_split(frame, days, cfg):
    # `frame` is already preprocessed; slicing it keeps every budget on the same
    # cleaned rows, so budgets differ only in how much history they include.
    raw = frame
    start, last = cgm_bounds(raw)
    end = last + pd.Timedelta(minutes=cfg['sampling_minutes'])
    if days != 'full':
        requested = end - pd.Timedelta(days=float(days))
        if requested < start:
            raise ValueError(f'History shorter than requested {days} days')
        start = requested
    # All budgets end together to avoid conflating data volume with recency.
    selected = raw.loc[(raw.index >= start) & (raw.index < end)].copy()
    boundary = start + (end - start) * (1 - cfg['validation_fraction'])
    train = selected.loc[selected.index < boundary].copy()
    val = selected.loc[selected.index >= boundary].copy()
    if train.empty or val.empty:
        raise ValueError('Empty chronological training or validation segment')
    return train, val, dict(history_start=str(start), history_end_exclusive=str(end),
                           validation_start=str(boundary), budget_days=days)


def fit_scaler(frame):
    values = frame.to_numpy(float)
    mean, sd = np.nanmean(values, axis=0), np.nanstd(values, axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(sd).all():
        raise ValueError('Feature has no valid training observations')
    return mean, np.where(sd > 1e-8, sd, 1.)


def windows(frame, cfg, mean, sd):
    a = frame.to_numpy(np.float32)
    timestamps = frame.index.to_numpy(dtype='datetime64[ns]')
    w, h = cfg['window'], cfg['horizon_steps']
    length = w + h
    if len(a) < length:
        return None
    valid = np.isfinite(a).all(axis=1)
    # Prefix sums test every adjacent interval AND all rows in the window.
    invalid = np.r_[0, np.cumsum(~valid)]
    gap = np.r_[0, np.diff(timestamps) != np.timedelta64(cfg['sampling_minutes'], 'm')]
    gaps = np.r_[0, np.cumsum(gap)]
    starts = np.arange(len(a) - length + 1)
    good = ((invalid[starts+length] - invalid[starts]) == 0) & (
        (gaps[starts+length] - gaps[starts+1]) == 0)
    starts = starts[good]
    if not len(starts):
        return None
    z = ((a - mean) / sd).astype(np.float32)
    x = np.stack([z[i:i+w] for i in starts])
    y = np.stack([z[i+w:i+length, 0] for i in starts])
    return dict(dataset=TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                timestamps=timestamps[starts+length-1],
                truth=a[starts+length-1, 0].astype(float),
                persistence=a[starts+w-1, 0].astype(float))


def prepare_segments(clean_train, clean_test, days, cfg):
    train, val, audit = budget_split(clean_train, days, cfg)
    if train.empty or val.empty:
        raise ValueError('Empty preprocessed segment')
    mean, sd = fit_scaler(train)
    train_w, val_w = windows(train, cfg, mean, sd), windows(val, cfg, mean, sd)
    # Report the counts: a budget that is short by one window and a budget with
    # a sensor outage covering the whole slice need different answers.
    n_train_w = len(train_w['dataset']) if train_w else 0
    n_val_w = len(val_w['dataset']) if val_w else 0
    if n_train_w < cfg['min_train_windows']:
        raise ValueError(f"Too few contiguous training windows: {n_train_w} < "
                         f"{cfg['min_train_windows']} (budget {days})")
    if n_val_w < cfg['min_validation_windows']:
        raise ValueError(f"Too few contiguous validation windows: {n_val_w} < "
                         f"{cfg['min_validation_windows']} (budget {days})")
    audit.update(n_train=len(train_w['dataset']), n_validation=len(val_w['dataset']),
                 normalization_mean=mean.tolist(), normalization_sd=sd.tolist())
    result = dict(train=train_w, val=val_w, mean=mean, sd=sd, audit=audit)
    if clean_test is not None:
        test = clean_test
        if train.index.max() >= val.index.min() or val.index.max() >= test.index.min():
            raise ValueError('Train/validation/test chronology violated')
        test_w = windows(test, cfg, mean, sd)
        n_test_w = len(test_w['dataset']) if test_w else 0
        if n_test_w < cfg['min_test_windows']:
            raise ValueError(f"Too few contiguous test windows: {n_test_w} < "
                             f"{cfg['min_test_windows']}")
        audit.update(n_test=len(test_w['dataset']), test_start=str(test.index.min()),
                     test_end=str(test.index.max()))
        result['test'] = test_w
    return result


class ForecastNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cls = {'gru': nn.GRU, 'lstm': nn.LSTM, 'rnn': nn.RNN}[cfg['model']]
        self.encoder = cls(4, cfg['hidden_size'], cfg['layers'], batch_first=True,
                          dropout=cfg['dropout'] if cfg['layers'] > 1 else 0.)
        self.head = nn.Linear(cfg['hidden_size'], cfg['horizon_steps'])

    def forward(self, x):
        sequence, _ = self.encoder(x)
        return self.head(sequence[:, -1])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fit(model, training, validation, cfg, device, seed, epochs, patience):
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(training, batch_size=cfg['batch_size'], shuffle=True,
                       generator=generator, pin_memory=device.startswith('cuda'))
    val = DataLoader(validation, batch_size=cfg['batch_size'], shuffle=False)
    model.to(device)
    # Always fresh, including fine-tuning: no optimizer state crosses stages.
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=1)
    best, stale, state = float('inf'), 0, None
    history = []
    for epoch in range(epochs):
        model.train()
        total, count = 0., 0
        for x, y in train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.item() * len(y)
            count += len(y)
        model.eval()
        vsum, vn = 0., 0
        with torch.no_grad():
            for x, y in val:
                pred = model(x.to(device))
                loss = nn.functional.mse_loss(pred, y.to(device))
                vsum += loss.item() * len(y)
                vn += len(y)
        vl = vsum / vn
        if not np.isfinite(vl):
            raise ValueError('Nonfinite validation loss')
        scheduler.step(vl)
        history.append(dict(epoch=epoch+1, train_mse=total/count, validation_mse=vl))
        print(f'    epoch {epoch+1}/{epochs}: val MSE {vl:.5f}', flush=True)
        if vl < best:
            best, stale = vl, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(state)
    return history


def evaluate(model, pack, cfg, device):
    model.eval()
    pred = []
    with torch.no_grad():
        for x, _ in DataLoader(pack['test']['dataset'], batch_size=cfg['batch_size']):
            pred.extend(model(x.to(device))[:, -1].cpu().numpy())
    pred = np.array(pred)*pack['sd'][0] + pack['mean'][0]
    truth = pack['test']['truth']
    diff = pred - truth
    persistence = pack['test']['persistence']
    metrics = dict(mae=float(abs(diff).mean()), rmse=float(np.sqrt(np.mean(diff**2))),
                   persistence_mae=float(abs(persistence-truth).mean()),
                   persistence_rmse=float(np.sqrt(np.mean((persistence-truth)**2))), n_test=len(truth))
    table = pd.DataFrame(dict(target_timestamp=pack['test']['timestamps'], truth=truth,
                             prediction=pred, persistence=persistence))
    return metrics, table


def make_protocol(data_root, support, cfg):
    files = []
    for release, patients in COHORT.items():
        for pid in patients:
            if pid not in cfg['patients']:
                continue
            for mode in ['train', 'test']:
                p = Path(data_root)/'raw'/'ohiot1dm'/release/mode/f'{pid}-ws-{mode}ing.xml'
                if not p.is_file():
                    raise FileNotFoundError(p)
                files.append(dict(path=str(p.relative_to(data_root)), sha256=digest(p)))
    return dict(config=cfg, input_files=files, torch_version=torch.__version__,
                numpy_version=np.__version__, pandas_version=pd.__version__,
                sources={name: digest(Path(support)/name) for name in ['loaders.py', 'preprocessors.py']},
                runner_sha256=digest(__file__),
                design='most recent target-history budget; chronological validation; fixed official test; '
                       'leave-target-out sources; preprocessing applied once per record before slicing')


def run(data_root, support, cfg, local_root, drive_root, device='cuda'):
    cfg = copy.deepcopy(cfg)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime in Colab before training.')
    if cfg['model'] not in ['gru', 'lstm', 'rnn'] or cfg['sampling_minutes'] != 5:
        raise ValueError('Supported models: GRU/LSTM/RNN; this protocol uses five-minute data.')
    if len(cfg['patients']) < 2 or len(set(cfg['patients'])) != len(cfg['patients']):
        raise ValueError('Need distinct target/source patients')
    if 'full' not in cfg['budgets_days'] or len(set(cfg['budgets_days'])) != len(cfg['budgets_days']):
        raise ValueError('Include one full-history control and distinct budgets')
    protocol = make_protocol(data_root, support, cfg)
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
    name = ('SMOKE_' if cfg['smoke'] else '') + f"{cfg['model']}_{5*cfg['horizon_steps']}min_{fingerprint}"
    local, remote = Path(local_root)/name, Path(drive_root)/name
    local.mkdir(parents=True, exist_ok=True)
    remote.mkdir(parents=True, exist_ok=True)
    # Restore only this exact data/config/code fingerprint. No stale result reuse.
    shutil.copytree(remote, local, dirs_exist_ok=True)
    atomic_json(local/'protocol.json', protocol)
    shutil.copy2(local/'protocol.json', remote/'protocol.json')
    loader, prep = load_support(support)
    raw = {}
    for pid in cfg['patients']:
        release = next(v for v, ids in COHORT.items() if pid in ids)
        raw[pid] = {}
        for mode in ['train', 'test']:
            with contextlib.redirect_stdout(io.StringIO()):
                raw[pid][mode] = loader.load_ohiot1dm_data(
                    str(data_root), patient_ids=[pid], mode=mode, version=release)[pid]
            raw[pid][mode].index = pd.to_datetime(raw[pid][mode].index)
    # Preflight EVERY budget before spending GPU time. Eligibility cannot be
    # selected by observed performance. Detailed errors are saved, then fail.
    source, audits, errors, clean = {}, [], [], {}
    print('Preflighting every patient/budget and source split...', flush=True)
    for pid in cfg['patients']:
        try:
            # Preprocess each record once; every budget then slices these rows.
            clean[pid] = {mode: clean_frame(raw[pid][mode], prep) for mode in ['train', 'test']}
            source[pid] = prepare_segments(clean[pid]['train'], None, 'full', cfg)
            reference = None
            for budget in cfg['budgets_days']:
                pack = prepare_segments(clean[pid]['train'], clean[pid]['test'], budget, cfg)
                current = pack['test']['timestamps']
                if reference is not None and not np.array_equal(current, reference):
                    raise ValueError('Test windows change across budgets')
                reference = current
                audits.append(dict(patient_id=pid, **pack['audit']))
        except Exception as exc:
            errors.append(dict(patient_id=pid, error=str(exc)))
    atomic_json(local/'preflight.json', dict(audits=audits, errors=errors))
    shutil.copy2(local/'preflight.json', remote/'preflight.json')
    if errors:
        raise ValueError(f'Preflight failed; inspect {remote}/preflight.json. No model trained. {errors}')
    print(f'Preflight passed. Outputs: {remote}', flush=True)
    for seed in cfg['seeds']:
        for pid in cfg['patients']:
            source_dir = local/'pretraining'/f'patient{pid}_seed{seed}'
            source_remote = remote/'pretraining'/source_dir.name
            source_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = source_dir/'weights.pt'
            complete_source = source_dir/'complete.json'
            valid_source = (complete_source.is_file() and checkpoint.is_file() and
                            json.loads(complete_source.read_text()).get('checkpoint_sha256') == digest(checkpoint))
            if not valid_source:
                print(f'Pretrain excluding patient {pid}, seed {seed}', flush=True)
                seed_everything(seed)
                model = ForecastNet(cfg)
                donors = [q for q in cfg['patients'] if q != pid]
                history = fit(model, ConcatDataset([source[q]['train']['dataset'] for q in donors]),
                              ConcatDataset([source[q]['val']['dataset'] for q in donors]),
                              cfg, device, seed, cfg['source_epochs'], cfg['source_patience'])
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, checkpoint)
                atomic_json(complete_source, dict(donors=donors, target_excluded=pid, history=history,
                                                  checkpoint_sha256=digest(checkpoint)))
                source_remote.mkdir(parents=True, exist_ok=True)
                shutil.copy2(checkpoint, source_remote/'weights.pt')
                shutil.copy2(complete_source, source_remote/'complete.json')
                del model
            expected_hash = json.loads(complete_source.read_text())['checkpoint_sha256']
            if digest(checkpoint) != expected_hash:
                raise ValueError('Cached source checkpoint hash mismatch')
            for budget in cfg['budgets_days']:
                job = local/'jobs'/f'patient{pid}_seed{seed}_days{budget}'
                target = remote/'jobs'/job.name
                done = job/'complete.json'
                if done.is_file():
                    saved = json.loads(done.read_text())
                    hashes = saved.get('files_sha256', {})
                    if hashes and all((job/name).is_file() and digest(job/name) == value
                                      for name, value in hashes.items()):
                        target.mkdir(parents=True, exist_ok=True)
                        for name in hashes:
                            shutil.copy2(job/name, target/name)
                        shutil.copy2(done, target/'complete.json')
                        print(f'Skip completed {job.name}', flush=True)
                        continue
                job.mkdir(parents=True, exist_ok=True)
                pack = prepare_segments(clean[pid]['train'], clean[pid]['test'], budget, cfg)
                row = dict(patient_id=pid, seed=seed, **pack['audit'])
                start_time = time.perf_counter()
                for mode in ['regular', 'transfer']:
                    print(f'{job.name}: {mode}', flush=True)
                    seed_everything(seed)
                    model = ForecastNet(cfg)
                    if mode == 'transfer':
                        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
                    history = fit(model, pack['train']['dataset'], pack['val']['dataset'], cfg,
                                  device, seed, cfg['target_epochs'], cfg['target_patience'])
                    metrics, predictions = evaluate(model, pack, cfg, device)
                    row[mode] = metrics
                    atomic_json(job/f'{mode}_history.json', history)
                    predictions.to_csv(job/f'{mode}_predictions.csv', index=False)
                    del model
                row['seconds'] = time.perf_counter() - start_time
                row['files_sha256'] = {p.name: digest(p) for p in job.iterdir()
                                      if p.name.endswith(('_history.json', '_predictions.csv'))}
                # Remote completion marker is copied LAST: interruption cannot
                # make an incomplete remote job look complete on next runtime.
                target.mkdir(parents=True, exist_ok=True)
                for path in job.iterdir():
                    if path.name != 'complete.json':
                        shutil.copy2(path, target/path.name)
                atomic_json(done, row)
                shutil.copy2(done, target/'complete.json')
                print(f'Completed and mirrored {job.name}: {row["seconds"]/60:.1f} min', flush=True)
    summarize(local, remote, cfg)
    return local, remote


def summarize(local, remote, cfg):
    from scipy.stats import wilcoxon
    rows = [json.loads(p.read_text()) for p in sorted((Path(local)/'jobs').glob('*/complete.json'))]
    expected = {(p, s, str(b)) for p in cfg['patients'] for s in cfg['seeds'] for b in cfg['budgets_days']}
    got = {(r['patient_id'], r['seed'], str(r['budget_days'])) for r in rows}
    if got != expected or len(rows) != len(expected):
        raise ValueError('Incomplete or duplicate cohort: summary withheld until all paired jobs finish')
    flat = []
    for r in rows:
        flat.append(dict(patient_id=r['patient_id'], seed=r['seed'], budget_days=str(r['budget_days']),
                         n_train=r['n_train'], n_validation=r['n_validation'], n_test=r['n_test'],
                         **{f'{mode}_{metric}': r[mode][metric] for mode in ['regular', 'transfer']
                            for metric in ['mae', 'rmse', 'persistence_mae', 'persistence_rmse']}))
    frame = pd.DataFrame(flat)
    np.testing.assert_allclose(frame.regular_persistence_mae, frame.transfer_persistence_mae)
    if (frame.groupby('patient_id').regular_persistence_mae.nunique() != 1).any():
        raise ValueError('Persistence/test values differ across budgets/seeds')
    out = Path(local)/'summary'
    out.mkdir(exist_ok=True)
    frame.to_csv(out/'patient_seed_metrics.csv', index=False)
    rng = np.random.default_rng(cfg['analysis_seed'])
    n, k = len(cfg['patients']), len(cfg['seeds'])
    reps = cfg['bootstrap_replicates']
    pi = rng.integers(0, n, size=(reps, n))
    si = rng.integers(0, k, size=(reps, k))
    summaries = []
    def array(budget, mode, metric):
        return frame[frame.budget_days == str(budget)].pivot(index='patient_id', columns='seed',
            values=f'{mode}_{metric}').loc[cfg['patients'], cfg['seeds']].to_numpy()
    for metric in ['mae', 'rmse']:
        full = array('full', 'regular', metric) - array('full', 'transfer', metric)
        for budget in cfg['budgets_days']:
            rl, tl = array(budget, 'regular', metric), array(budget, 'transfer', metric)
            diff = rl-tl
            sampled_rl = rl[pi[:, :, None], si[:, None, :]].mean(axis=(1, 2))
            sampled_diff = diff[pi[:, :, None], si[:, None, :]].mean(axis=(1, 2))
            full_diff = full[pi[:, :, None], si[:, None, :]].mean(axis=(1, 2))
            percent = 100*sampled_diff/sampled_rl
            per_patient = diff.mean(axis=1)
            p = 1. if np.allclose(per_patient, 0) else float(wilcoxon(per_patient).pvalue)
            summaries.append(dict(metric=metric, budget_days=str(budget),
                regular_mean=float(rl.mean()), transfer_mean=float(tl.mean()),
                benefit=float(diff.mean()), benefit_ci_low=float(np.quantile(sampled_diff,.025)),
                benefit_ci_high=float(np.quantile(sampled_diff,.975)),
                relative_benefit_pct=float(100*diff.mean()/rl.mean()),
                relative_ci_low=float(np.quantile(percent,.025)), relative_ci_high=float(np.quantile(percent,.975)),
                additional_benefit_vs_full=float((diff-full).mean()),
                additional_ci_low=float(np.quantile(sampled_diff-full_diff,.025)),
                additional_ci_high=float(np.quantile(sampled_diff-full_diff,.975)),
                wilcoxon_p=p, n_patients=n, n_seeds=k))
    table = pd.DataFrame(summaries)
    ps = table.wilcoxon_p.to_numpy(); order = np.argsort(ps); q = np.empty(len(ps))
    q[order] = np.minimum.accumulate((ps[order]*len(ps)/np.arange(1,len(ps)+1))[::-1])[::-1]
    table['wilcoxon_bh_q'] = np.minimum(q,1)
    table.to_csv(out/'budget_summary.csv', index=False)
    text = ['# Cold-start history-budget experiment', '',
        'SMOKE RUN — not article evidence.' if cfg['smoke'] else 'Exploratory follow-up; no outcome-driven budget selection.',
        '', f"Model: {cfg['model'].upper()}, horizon: {5*cfg['horizon_steps']} minutes; {n} patients, {k} seeds.",
        'Budgets use the most recent available training history, including the chronological validation portion. Official test windows are fixed across all budgets.',
        'Positive benefit = RL error minus TL error. Percent benefit is the reduction in equal-patient cohort mean error, not a pooled-window improvement.',
        '95% intervals jointly resample whole patients and shared training seeds; additional-benefit intervals use the same draws for the budget and full-history control. Intervals are pointwise, not simultaneous.',
        'Wilcoxon tests use patient cross-seed means; BH spans both metrics and every budget in this run. Different architectures/horizons run separately would require a broader family before combined claims.',
        '', '## Interpretation', '',
        'The primary comparison is MAE at 3 days versus this experiment\'s full-history control. A larger advantage is a hypothesis, not guaranteed. Report all budgets, including null or adverse effects. Inspect absolute RL/TL errors and persistence as well as percentage gains.',
        'This is simulated limited-history personalization with a retrospective source library, not a prospectively recruited incident-patient study. Other patients\' source records are assumed available; their calendar dates are not restricted to each target\'s test date.',
        'Chronological validation, gap-checked windows, source-only pretraining validation, and matched 50-epoch target caps differ from the article\'s original schedule. Compare budgets within this new experiment; do not replace the published full-history headline with its best budget.',
        'The budgets share the same final training time, controlling history recency. Test separation/gaps follow the official Ohio split. This experiment cannot explain why earlier submitted numbers changed.',
        '', 'See budget_summary.csv, patient_seed_metrics.csv, preflight.json and protocol.json. Per-job predictions and training histories are retained.', '']
    (out/'REPORT.md').write_text('\n'.join(text))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,metric in zip(axes,['mae','rmse']):
        t = table[table.metric == metric].set_index('budget_days').loc[list(map(str,cfg['budgets_days']))]
        x = np.arange(len(t))
        ax.plot(x,t.relative_benefit_pct,marker='o')
        ax.vlines(x,t.relative_ci_low,t.relative_ci_high)
        ax.axhline(0,color='gray',lw=.8)
        ax.set(xticks=x,xticklabels=t.index,xlabel='Target history (days; full = all)',
               ylabel=f'{metric.upper()} reduction (%)',title=f'{metric.upper()}: joint patient/seed 95% intervals')
    fig.savefig(out/'history_budget_curve.png',dpi=180)
    plt.close(fig)
    shutil.copytree(out,Path(remote)/'summary',dirs_exist_ok=True)
    print(table.to_string(index=False),flush=True)
