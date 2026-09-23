"""Full Cui-inspired TL / RL schedule grid, resumable by completed mode/seed."""
from pathlib import Path
from dataclasses import replace
import contextlib, copy, hashlib, io, json, os, shutil, subprocess, sys
import numpy as np
import pandas as pd
import torch
import yaml
from benchmark.configs import load_config
from benchmark.experiments import configured as engine

ARMS = {'patience5': (200, 5), 'patience15': (200, 15), 'fixed20': (20, 20)}
STAGES = ['zone-d','error-range','rapid-change','figures','glycemic','tsne',
          'persistence','shift','transfer','stability','horizons','compare','summary']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)

def complete(folder):
    folder=Path(folder);marker=folder/'complete.json'
    if not marker.is_file():return False
    try:
        hashes=json.loads(marker.read_text())['files']
        return bool(hashes) and all((folder/p).is_file() and sha(folder/p)==v for p,v in hashes.items())
    except (ValueError,KeyError,OSError):return False

def seal(folder):
    write(Path(folder)/'complete.json',{'files':{str(p.relative_to(folder)):sha(p)
          for p in sorted(Path(folder).rglob('*')) if p.is_file() and p.name!='complete.json'}})

def mirror(source,target):
    source,target=Path(source),Path(target);target.mkdir(parents=True,exist_ok=True)
    # Publish completion marker after all payloads.
    for p in source.rglob('*'):
        if p.is_file() and p.name!='complete.json':
            q=target/p.relative_to(source);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    if (source/'complete.json').exists():shutil.copy2(source/'complete.json',target/'complete.json')

def setup(repo, drive, models, horizons, seeds, device='cuda', smoke=False):
    repo=Path(repo).resolve();os.chdir(repo)
    configs={}
    for model in models:
        for h in horizons:
            c=load_config(repo/f'configs/full_{model}_{h}min.yaml')
            assert c.preprocessing.window_size==12 and c.preprocessing.prediction_horizon==h//5
            assert not c.preprocessing.unimodal and not c.preprocessing.include_feature_engineering
            assert len(c.data.patient_ids())==12
            c=replace(c,model=replace(c.model,architecture={**c.model.architecture,'hidden_size':128,'num_layers':2,'dropout':.2}),
                training=replace(c.training,seeds=seeds,device=device,mode='both',epochs=200,
                    batch_size=16,learning_rate=.0003,finetune_learning_rate=.00005,
                    pretrain_epochs=10,finetune_epochs=10,early_stopping_patience=5,
                    transfer_early_stopping_patience=20,regular_schedule='single_stage'),
                output=replace(c.output,save_predictions=True,save_model=False,generate_plots=False,export_format=['json','csv']))
            if smoke:
                c=replace(c,data=replace(c.data,patients=[559,563]),
                    model=replace(c.model,architecture={**c.model.architecture,'hidden_size':4,'num_layers':1,'dropout':0.}))
            configs[f'{model}_{h}']=c
    inputs={}
    for c in configs.values():
        for pid in c.data.patient_ids():
            for mode in ['train','test']:
                p=Path(c.data.root)/'raw/ohiot1dm'/c.data.version_for_patient(pid)/mode/f'{pid}-ws-{mode}ing.xml'
                inputs[str(p)]=sha(p)
    paths=list((repo/'benchmark').rglob('*.py'))+list((repo/'RUN').rglob('*.py'))+list((repo/'RUN').glob('*.sh'))
    protocol=dict(configs={k:v.to_dict() for k,v in configs.items()},arms=ARMS,smoke=smoke,
        inputs=inputs,sources={str(p.relative_to(repo)):sha(p) for p in sorted(paths)},
        torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        seed_rule='seed*100000+patient_id',version=1)
    key=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()[:16]
    name=('SMOKE_' if smoke else '')+key
    local=repo/'results/cui_full_grid'/name;remote=Path(drive)/'cui_full_grid'/name
    local.mkdir(parents=True,exist_ok=True);remote.mkdir(parents=True,exist_ok=True)
    for part in ['jobs','analysis']:
        if (remote/part).exists():shutil.copytree(remote/part,local/part,dirs_exist_ok=True)
    write(local/'protocol.json',protocol);shutil.copy2(local/'protocol.json',remote/'protocol.json')
    return configs,local,remote

def metrics(folder, config, mode):
    entries=json.loads((folder/'metrics.json').read_text())
    if set(map(int,entries))!=set(config.data.patient_ids()):raise ValueError('Incomplete patient cohort')
    for pid,e in entries.items():
        if not np.isfinite(e['mae']):raise ValueError('Nonfinite MAE')
        if not (folder/f'patient_{pid}'/f'{config.model.type.upper()}_predictions.csv').is_file():
            raise ValueError('Missing predictions')
        h=e['training_history']
        if mode=='transfer' and (h['pretrain_history']['epochs_completed']!=10 or h['finetune_history']['epochs_completed']!=10):
            raise ValueError('TL must complete 10+10 epochs')
        if mode=='regular' and config.training.epochs==20 and h['epochs_completed']!=20:
            raise ValueError('Fixed-20 arm stopped early')
    return entries

class Tee:
    def __init__(self, *streams):self.streams=streams
    def write(self, value):
        for stream in self.streams:stream.write(value);stream.flush()
        return len(value)
    def flush(self):
        for stream in self.streams:stream.flush()

def train(configs,local,remote):
    first=next(iter(configs.values()))
    if first.training.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Select GPU')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    original=engine._make_loaders
    def paired(config,mode,seed,patient_id,frames):
        engine._seed_everything(seed*100000+patient_id)
        return original(config,mode,seed,patient_id,frames)
    engine._make_loaders=paired
    try:
        print('Loading and preprocessing the cohort once...',flush=True)
        with contextlib.redirect_stdout(io.StringIO()):frames=engine._load_patient_frames(first,first.data.patient_ids())
        for seed in first.training.seeds:
            for cell,c in configs.items():
                for arm in ['transfer',*ARMS]:
                    mode='transfer' if arm=='transfer' else 'regular'
                    epochs,patience=(20,20) if mode=='transfer' else ARMS[arm]
                    cfg=replace(c,training=replace(c.training,mode=mode,seeds=[seed],epochs=epochs,early_stopping_patience=patience))
                    job=local/'jobs'/cell/arm/f'seed_{seed}';target=remote/job.relative_to(local)
                    if complete(job):
                        metrics(job,cfg,mode);print('Skip',cell,arm,seed,flush=True);continue
                    print('Train',cell,arm,seed,flush=True)
                    job.mkdir(parents=True,exist_ok=True)
                    try:
                        with (job/'training.log').open('w') as log, contextlib.redirect_stdout(Tee(sys.stdout,log)):
                            engine._run_mode(cfg,mode,seed,c.data.patient_ids(),frames,job)
                    finally:
                        target.mkdir(parents=True,exist_ok=True)
                        shutil.copy2(job/'training.log',target/'training.log')
                    metrics(job,cfg,mode);seal(job);mirror(job,target)
    finally:engine._make_loaders=original

def merge(configs,local,remote):
    for arm,(cap,pat) in ARMS.items():
        parents=[]
        for cell,c in configs.items():
            out=local/'paired'/arm/cell;out.mkdir(parents=True,exist_ok=True)
            cfg=replace(c,training=replace(c.training,epochs=cap,early_stopping_patience=pat))
            runs=[];sources={}
            for seed in c.training.seeds:
                for mode,source_arm in [('regular',arm),('transfer','transfer')]:
                    source=local/'jobs'/cell/source_arm/f'seed_{seed}'
                    if not complete(source):raise ValueError(f'Incomplete job: {source}')
                    entries=metrics(source,cfg,mode)
                    dest=out/mode/f'seed_{seed}';shutil.copytree(source,dest,dirs_exist_ok=True)
                    sources[f'{mode}/{seed}']=str(source)
                    runs.append(dict(mode=mode,seed=seed,results=entries))
            aggregates=engine._aggregate_runs(runs,cfg)
            write(out/'aggregate_metrics.json',aggregates)
            pd.DataFrame(engine._aggregate_rows(aggregates)).to_csv(out/'aggregate_metrics.csv',index=False)
            (out/'resolved_config.yaml').write_text(yaml.safe_dump(cfg.to_dict(),sort_keys=False))
            write(out/'tracking.json',dict(experiment_id=cell,config=cfg.to_dict(),status='merged',
                data_params=dict(dataset=cfg.data.dataset,patient_ids=cfg.data.patient_ids(),
                    version=cfg.data.version,sequence_length=cfg.preprocessing.window_size,
                    prediction_horizon=cfg.preprocessing.prediction_horizon,
                    sampling_rate_minutes=cfg.preprocessing.sampling_rate,seeds=cfg.training.seeds,modes=['regular','transfer'])))
            write(out/'merge_manifest.json',dict(sources=sources,shared_TL=True))
            model,h=cell.rsplit('_',1);parents.append(f'{model} {h} {out}\n')
        folder=local/'logs'/arm;folder.mkdir(parents=True,exist_ok=True)
        (folder/'parents.txt').write_text(''.join(parents))
    mirror(local/'paired',remote/'paired');mirror(local/'logs',remote/'logs')

def summarize(configs,local,remote,reps=20000):
    from scipy.stats import wilcoxon
    rng=np.random.default_rng(42);first=next(iter(configs.values()))
    n,k=len(first.data.patient_ids()),len(first.training.seeds)
    pi=rng.integers(n,size=(reps,n));si=rng.integers(k,size=(reps,k))
    def sample(a):return a[pi[:,:,None],si[:,None,:]].mean(axis=(1,2))
    rows=[];diagnostics=[]
    for cell,c in configs.items():
        def read(arm):
            return [json.loads((local/'jobs'/cell/arm/f'seed_{s}/metrics.json').read_text()) for s in c.training.seeds]
        tl=read('transfer');rl0=read('patience5');ids=list(map(str,c.data.patient_ids()))
        for arm in ARMS:
            rl=read(arm)
            for j,seed in enumerate(c.training.seeds):
                for pid in ids:
                    a=rl0[j][pid]['training_history']['val_losses'];b=rl[j][pid]['training_history']['val_losses'];length=min(len(a),len(b))
                    if not length or not np.allclose(a[:length],b[:length],rtol=1e-5,atol=1e-7):raise ValueError(f'Unpaired RL trajectory {cell}/{arm}/{pid}/{seed}')
                    diagnostics.append(dict(cell=cell,arm=arm,patient_id=pid,seed=seed,epochs=len(b),best_epoch=int(np.argmin(b))+1))
            for metric in ['mae','rmse']:
                def arr(items):return np.array([[items[j][p][metric] for j in range(k)] for p in ids])
                r,t,base=arr(rl),arr(tl),arr(rl0)
                d=r-t;boot=sample(d);pct=100*boot/sample(r);change=sample(r-base)
                pp=d.mean(axis=1);pv=1. if np.allclose(pp,0) else float(wilcoxon(pp).pvalue)
                rows.append(dict(cell=cell,arm=arm,metric=metric,regular_mean=r.mean(),transfer_mean=t.mean(),
                    benefit=d.mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),
                    benefit_pct=100*d.mean()/r.mean(),pct_low=np.quantile(pct,.025),pct_high=np.quantile(pct,.975),
                    change_vs_patience5=(r-base).mean(),change_low=np.quantile(change,.025),change_high=np.quantile(change,.975),wilcoxon_p=pv))
    df=pd.DataFrame(rows);pv=df.wilcoxon_p.to_numpy();order=np.argsort(pv);q=np.empty(len(pv))
    q[order]=np.minimum.accumulate((pv[order]*len(pv)/np.arange(1,len(pv)+1))[::-1])[::-1];df['global_bh_q']=np.minimum(q,1)
    out=local/'sensitivity';out.mkdir(exist_ok=True)
    df.to_csv(out/'full_grid_sensitivity.csv',index=False);pd.DataFrame(diagnostics).to_csv(out/'stopping_epochs.csv',index=False)
    mirror(out,remote/'sensitivity');return df

def analyze_stage(repo,configs,local,remote,arm,stage,reps=20000,permutations=10000):
    c=next(iter(configs.values()));out=local/'analysis'/arm;logs=local/'logs'/arm
    env=dict(os.environ,MODELS=' '.join(sorted({v.model.type for v in configs.values()})),
        HORIZONS=' '.join(map(str,sorted({v.preprocessing.prediction_horizon*5 for v in configs.values()}))),
        SEEDS=' '.join(map(str,c.training.seeds)),LOG_DIR=str(logs),ANALYSIS_DIR=str(out),
        DATA_ROOT=c.data.root,REPLICATES=str(reps),PERMUTATIONS=str(permutations),
        FIGURE_MODEL='gru' if any(v.model.type=='gru' for v in configs.values()) else c.model.type,
        FIGURE_HORIZON='30' if any(v.preprocessing.prediction_horizon==6 for v in configs.values()) else str(c.preprocessing.prediction_horizon*5),
        FIGURE_SEED=str(c.training.seeds[0]),PYTHONUNBUFFERED='1',
        PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH',''))
    token=hashlib.sha256(f'{stage}:{reps}:{permutations}'.encode()).hexdigest()[:12]
    marker=out/f'.stage_{token}.json'
    if marker.exists():
        saved=json.loads(marker.read_text())
        if saved.get('files') and all((out/p).is_file() and sha(out/p)==h for p,h in saved['files'].items()):
            print('Skip analysis',arm,stage);return
    script='RUN/run_dataset_analysis.sh' if stage=='dataset' else 'RUN/run_experiments_analysis.sh'
    arg='all' if stage=='dataset' else stage
    try:
        subprocess.run(['bash',script,arg],cwd=repo,env=env,check=True)
        write(marker,dict(stage=stage,replicates=reps,permutations=permutations,
            files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and not p.name.startswith('.stage_')}))
    finally:
        if out.exists():mirror(out,remote/'analysis'/arm)
        mirror(logs,remote/'logs'/arm)
