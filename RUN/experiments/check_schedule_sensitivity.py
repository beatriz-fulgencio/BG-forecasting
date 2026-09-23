"""Offline checks for sensitivity notebook orchestration and paired analysis."""
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import copy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import yaml

NOTEBOOK = json.loads((ROOT / 'notebooks/run_schedule_sensitivity_on_colab.ipynb').read_text())
def cell(i):
    return ''.join(NOTEBOOK['cells'][i]['source'])

class SensitivityChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        (root/'configs').mkdir()
        (root/'configs/full_gru_30min.yaml').write_text((ROOT/'configs/full_gru_30min.yaml').read_text())
        self.env = dict(REPO_DIR=str(root), BASE_CONFIG=root/'configs/full_gru_30min.yaml',
                        SEEDS=[41,42,43], DEVICE_CFG='cpu', DRIVE_RESULTS=str(root/'drive'),
                        COHORT={'2018':[559,563,570,575,588,591], '2020':[540,544,552,567,584,596]},
                        dst=root/'data', sha=lambda p: 'fixture', NOTEBOOK_CODE_SHA256='fixture')
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell(8), self.env)
        self.env.update(json=json, np=np, pd=pd, yaml=yaml)
        import pathlib, shutil
        self.env.update(pathlib=pathlib, shutil=shutil)
        tree = ast.parse(cell(10))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) or
                 isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                     ['EXPECTED_PATIENTS','LAUNCH'] for t in n.targets)]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<notebook helpers>','exec'), self.env)
        self.env['EXP_DIR'] = self.env['RUN_LOCAL']/'experiments'
        arms = {}
        for j,(name,arm) in enumerate(self.env['ARMS'].items()):
            parent=self.env['EXP_DIR']/('experiment_'+str(j)); parent.mkdir(parents=True)
            (parent/'resolved_config.yaml').write_text(yaml.safe_dump(self.env['EXPECTED_CONFIGS'][name]))
            (parent/'aggregate_metrics.json').write_text('{}')
            for mode in (['regular','transfer'] if j==0 else ['regular']):
                for seed in [41,42,43]:
                    epochs=[10,25,20][j]
                    history=dict(epochs_completed=epochs,val_losses=[1/(x+1) for x in range(epochs)])
                    if mode=='transfer':
                        history=dict(epochs_completed=20,pretrain_history={'epochs_completed':10},
                                     finetune_history={'epochs_completed':10})
                    entries={str(pid):dict(mae=12.+(0 if mode=='transfer' else 1-j*.1),
                                          training_history=history)
                             for pid in self.env['EXPECTED_PATIENTS']}
                    folder=parent/mode/f'seed_{seed}';folder.mkdir(parents=True)
                    (folder/'metrics.json').write_text(json.dumps(entries))
            arms[name]=parent
        self.env['arm_dirs']=arms

    def test_configs_and_analysis(self):
        for name in self.env['ARMS']:
            self.assertEqual(self.env['find_arm'](name), self.env['arm_dirs'][name])
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cell(12),self.env)
            exec(cell(14),self.env)
        t=self.env['table']
        np.testing.assert_allclose(t.change_vs_arm0,[0,-.1,-.2],atol=1e-12)
        self.assertIn('wilcoxon_bh_q', t)
        self.assertEqual(len(self.env['epoch_table']),4)

    def test_real_training_keeps_later_patient_paired(self):
        import torch
        import benchmark.experiments.configured as configured
        from benchmark.configs import load_config
        torch.set_num_threads(1)
        original = configured._make_loaders
        # Execute the exact notebook launcher setup without starting its CLI.
        exec(self.env['LAUNCH'].split("sys.argv =")[0], {})
        self.addCleanup(setattr, configured, '_make_loaders', original)
        frames = {}
        for pid in [559, 563]:
            frames[pid] = {}
            for mode in ['train', 'test']:
                x = np.arange(96)
                frames[pid][mode] = pd.DataFrame(dict(glucose=120+20*np.sin(x/8),
                    basal=np.ones(96), bolus=(x%24==0).astype(float), carbs=(x%24==0)*15.),
                    index=pd.date_range('2020-01-01' if mode=='train' else '2020-02-01', periods=96, freq='5min'))
        histories = []
        for epochs in [2, 4]:
            cfg = copy.deepcopy(self.env['base'])
            cfg['data']['patients'] = [559,563]
            cfg['training'].update(mode='regular', seeds=[41], device='cpu', epochs=epochs)
            cfg['model']['architecture'].update(hidden_size=4, num_layers=1, dropout=0.)
            cfg['output'].update(save_predictions=False, generate_plots=False)
            cfg['evaluation']['metrics'] = ['mae','rmse']
            path=Path(self.tmp.name)/f'tiny{epochs}.yaml';path.write_text(yaml.safe_dump(cfg))
            folder=Path(self.tmp.name)/f'train{epochs}'
            with contextlib.redirect_stdout(io.StringIO()):
                configured._run_mode(load_config(path),'regular',41,[559,563],frames,folder)
            histories.append(json.loads((folder/'metrics.json').read_text()))
        for pid in ['559','563']:
            np.testing.assert_array_equal(histories[0][pid]['training_history']['val_losses'],
                                          histories[1][pid]['training_history']['val_losses'][:2])

    def test_missing_patient_rejected(self):
        name=next(iter(self.env['arm_dirs']));parent=self.env['arm_dirs'][name]
        path=parent/'regular/seed_41/metrics.json'
        entries=json.loads(path.read_text());entries.pop(next(iter(entries)));path.write_text(json.dumps(entries))
        self.assertIsNone(self.env['find_arm'](name))
        with self.assertRaises(ValueError):self.env['checked_metrics'](parent,'regular',41)

    def test_stale_config_rejected(self):
        name=next(iter(self.env['arm_dirs']));path=self.env['arm_dirs'][name]/'resolved_config.yaml'
        cfg=yaml.safe_load(path.read_text());cfg['training']['epochs']=123
        path.write_text(yaml.safe_dump(cfg))
        self.assertIsNone(self.env['find_arm'](name))

    def test_unpaired_trajectories_rejected(self):
        parent=list(self.env['arm_dirs'].values())[1];path=parent/'regular/seed_41/metrics.json'
        entries=json.loads(path.read_text());entries[next(iter(entries))]['training_history']['val_losses'][0]=999
        path.write_text(json.dumps(entries))
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'trajectories not paired'):
            exec(cell(12),self.env)

if __name__=='__main__':unittest.main()
