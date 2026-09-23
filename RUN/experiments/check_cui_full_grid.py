"""Offline smoke test for training, resume, merge and analysis-reader compatibility."""
import contextlib,io,sys,tempfile,unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
import run_cui_full_grid as grid
from benchmark.configs import load_config
from benchmark.analysis.results_io import load_experiment_results

class FullGridCheck(unittest.TestCase):
    def test_end_to_end(self):
        torch.set_num_threads(1)
        c=load_config(ROOT/'configs/full_gru_30min.yaml')
        c=replace(c,data=replace(c.data,patients=[559,563]),
            training=replace(c.training,seeds=[41],device='cpu',regular_schedule='single_stage',
                learning_rate=.0003,finetune_learning_rate=.00005,transfer_early_stopping_patience=20),
            model=replace(c.model,architecture=dict(hidden_size=4,num_layers=1,dropout=0.,batch_first=True)),
            output=replace(c.output,save_predictions=True,save_model=False,generate_plots=False),
            evaluation=replace(c.evaluation,metrics=['mae','rmse']))
        frames={}
        for pid in c.data.patient_ids():
            frames[pid]={}
            for mode in ['train','test']:
                x=np.arange(64)
                frames[pid][mode]=pd.DataFrame(dict(glucose=np.round(120+20*np.sin(x/8)),basal=np.ones(64),
                    bolus=(x%20==0).astype(float),carbs=(x%20==0)*15.),
                    index=pd.date_range('2020-01-01' if mode=='train' else '2020-02-01',periods=64,freq='5min'))
        configs={'gru_30':c}
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            local=Path(tmp)/'local';remote=Path(tmp)/'drive';local.mkdir();remote.mkdir()
            with patch.object(grid.engine,'_load_patient_frames',return_value=frames),patch.dict(grid.ARMS,{'patience5':(2,5),'patience15':(4,15),'fixed20':(20,20)},clear=True):
                grid.train(configs,local,remote)
                self.assertEqual(len(list(remote.rglob('complete.json'))),4)
                with patch.object(grid.engine,'_run_mode',side_effect=AssertionError('Retrained completed job')):
                    grid.train(configs,local,remote)
                df=grid.summarize(configs,local,remote,reps=99)
                self.assertEqual(len(df),6)
                grid.merge(configs,local,remote)
                for arm in grid.ARMS:
                    for mode in ['regular','transfer']:
                        loaded=load_experiment_results(local/'paired'/arm/'gru_30',mode=mode,with_series=True)
                        self.assertIsNotNone(loaded)
                # Shared TL copies must be byte-identical in every arm.
                tl=[local/'paired'/a/'gru_30/transfer/seed_41/metrics.json' for a in grid.ARMS]
                self.assertEqual(len({grid.sha(p) for p in tl}),1)
                job=local/'jobs/gru_30/fixed20/seed_41'
                pred=next(job.glob('patient_*/*_predictions.csv'));pred.write_text('corrupt')
                self.assertFalse(grid.complete(job))

if __name__=='__main__':unittest.main()
