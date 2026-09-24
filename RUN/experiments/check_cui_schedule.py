"""Run: python RUN/experiments/check_cui_schedule.py (no pytest dependency)."""
import contextlib,io,json,sys,tempfile,unittest
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from benchmark.configs import load_config
from benchmark.experiments import configured
from benchmark.models.base_model import BasePyTorchBGModel

class CuiScheduleTests(unittest.TestCase):
    def test_all_configs(self):
        for model in ['gru','lstm','rnn']:
            for horizon in [15,30,45,60]:
                for suffix in ['', '_seed41','_seed42','_seed43']:
                    p=ROOT/'configs'/('seedwise' if suffix else '')/f'full_{model}_{horizon}min{suffix}.yaml'
                    c=load_config(p)
                    self.assertEqual(c.training.regular_schedule,'two_stage')
                    self.assertEqual((c.training.pretrain_epochs,c.training.finetune_epochs),(10,10))
                    self.assertEqual((c.training.learning_rate,c.training.finetune_learning_rate),(.0003,.00005))
                    self.assertEqual(c.training.batch_size,16)
                    self.assertGreaterEqual(c.training.early_stopping_patience,10)
        self.assertEqual(load_config(ROOT/'configs/example_experiment.yaml').training.regular_schedule,'single_stage')

    def test_real_two_stage_training(self):
        torch.set_num_threads(1)
        c=load_config(ROOT/'configs/full_gru_30min.yaml')
        c=replace(c,training=replace(c.training,device='cpu'),
                  model=replace(c.model,architecture=dict(hidden_size=4,num_layers=1,dropout=0.,batch_first=True)),
                  output=replace(c.output,save_predictions=False,generate_plots=False),
                  evaluation=replace(c.evaluation,metrics=['mae','rmse']))
        frames={}
        for pid in [559,563]:
            frames[pid]={}
            for mode in ['train','test']:
                x=np.arange(64)
                frames[pid][mode]=pd.DataFrame(dict(glucose=120+20*np.sin(x/8),basal=np.ones(64),
                    bolus=(x%20==0).astype(float),carbs=(x%20==0)*15.),
                    index=pd.date_range('2020-01-01' if mode=='train' else '2020-02-01',periods=64,freq='5min'))
        calls=[]
        original=BasePyTorchBGModel.fit
        def record(model,*args,**kwargs):
            self.assertIsNone(model.optimizer)
            self.assertEqual(kwargs['early_stopping_patience'],20)
            calls.append((kwargs['learning_rate'],len(kwargs['train_loader'].dataset)))
            return original(model,*args,**kwargs)
        BasePyTorchBGModel.fit=record
        try:
            with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
                for mode in ['regular','transfer']:
                    folder=Path(tmp)/mode
                    run_cfg = replace(c, training=replace(c.training, early_stopping_patience=5,
                                      transfer_early_stopping_patience=20)) if mode == 'transfer' else c
                    configured._run_mode(run_cfg,mode,41,[559],frames,folder)
                    history=json.loads((folder/'metrics.json').read_text())['559']['training_history']
                    self.assertEqual(history['epochs_completed'],20)
                    self.assertEqual(history['pretrain_history']['epochs_completed'],10)
                    self.assertEqual(history['finetune_history']['epochs_completed'],10)
        finally:BasePyTorchBGModel.fit=original
        self.assertEqual([v[0] for v in calls],[.0003,.00005,.0003,.00005])
        self.assertEqual(calls[0][1],calls[1][1])
        self.assertEqual(calls[1][1],calls[3][1])
        self.assertNotEqual(calls[2][1],calls[3][1])

if __name__=='__main__':unittest.main()
