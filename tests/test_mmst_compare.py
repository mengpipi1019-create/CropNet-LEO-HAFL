import contextlib
import io
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cropnet_leo.remote import sha
from research.joint_fusion import run_mmst_compare as compare
from research.joint_fusion import run_mmst_local as local
from research.joint_fusion import run_mmst_full_repaired as full
from research.joint_fusion import plot_mmst_compare as plots
from research.joint_fusion.mmst_io import write_json


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.fc=torch.nn.Linear(1,2);self.drop=torch.nn.Dropout(.2)
    def forward(self,x,ys=None,yl=None):return self.fc(self.drop(x))


class Data:
    def __init__(self,path,*args):
        self.root=Path(path)
        self.manifest={'evaluation':False,'records':[{'record':{'year':2021,'state':str(i//2),'fips':str(i)}} for i in range(8)]}
    def __len__(self):return 8
    def __call__(self,i):
        return torch.tensor([[i/8.]]),torch.zeros(1),torch.zeros(1),torch.tensor([[i/8.,1.+i/8.]])


class CompareTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required for command-entry rehearsal')
    def test_actual_main_entry_completes_all_nine_and_second_run_does_not_train_again(self):
        cfg={**local.read(compare.CONFIG),**self.config()};stats={'target':{'mean':[0.,0.],'std':[1.,1.]}}
        rows=[{'state':str(i//2),'fips':str(i)} for i in range(8)]
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            root=Path(folder);base=root/'base';base.mkdir();prepared=base/'prepared_train';prepared.mkdir()
            write_json(base/'config.json',cfg);write_json(prepared/'manifest.json',{'synthetic_qa_only':True})
            write_json(base/'download_manifest.json',{'repository':cfg['dataset_repository'],'revision':cfg['dataset_revision'],'include_evaluation':True,'files':[]})
            config_path=root/'config.json';write_json(config_path,cfg)
            evaluation=root/'evaluation_fixture';evaluation.mkdir();write_json(evaluation/'manifest.json',{'synthetic_qa_only':True})
            out=root/'comparison'
            with patch.object(compare,'BASE',base),patch.object(compare,'preflight',lambda *a:(cfg,Data(prepared),{})), \
                 patch.object(compare,'ensure_inputs',lambda *a:(evaluation,{'status':'completed','synthetic_qa_only':True})), \
                 patch.object(local,'CountyData',Data),patch.object(full,'fit_statistics',lambda _:stats), \
                 patch.object(full,'make_model',lambda _:Toy()),patch.object(full,'records',lambda _:rows), \
                 patch.object(local,'make_model',lambda _:Toy()),patch.object(local,'records',lambda *a:rows), \
                 patch.object(sys,'argv',['qa','--config',str(config_path),'--out',str(out),'--supervised']):
                compare.main()
                self.assertEqual(local.read(out/'status.json')['status'],'completed')
                self.assertEqual(len(local.read(out/'evaluation/metrics.json')),9)
                self.assertTrue((out/'evaluation/paper_table.pdf').exists())
                local.full.check_sources(out)
                with patch.object(compare,'run_pipeline',side_effect=AssertionError('Completed output must be reused')):
                    compare.main()
    def config(self):
        return {'epochs':4,'warmup_epochs':1,'decay_after_epoch':2,'minimum_lr_ratio':.01,
                'learning_rate':.01,'weight_decay':.01,'gradient_clip':5.,'county_batch_size':2,
                'checkpoint_every_epochs':1,'server_gain':compare.GAIN,'staleness_power':0,
                'logical_ring_delays':[1.,1.2,1.4,1.6],'max_ring_lead':2,'seeds':[10,20,30],
                'checkpoint_rule':'fixed final','clients_per_state':1,'train_year':2021,'evaluation_year':2022}

    def test_cache_guard_only_allows_frozen_balanced_aggregation_and_metadata(self):
        cfg=self.config();old={**cfg,'server_gain':1.5,'staleness_power':1,'checkpoint_every_epochs':10}
        compare.require_shared_inputs(cfg,old)
        for field,value in [('learning_rate',.2),('epochs',120),('train_year',2022),('server_gain',.9),('staleness_power',1)]:
            with self.assertRaises(ValueError):compare.require_shared_inputs({**cfg,field:value},old)

    def test_probe_preserves_mode_weights_and_rng_and_is_training_only(self):
        model=Toy();model.train();initial=compare.vector(model);before=torch.get_rng_state().clone()
        data=Data('.');data.stats={'target':{'mean':[0,0],'std':[1,1]}}
        result=compare.probe_measurement(model,initial,data,data.stats)
        torch.testing.assert_close(compare.vector(model),initial,rtol=0,atol=0)
        self.assertTrue(model.training);self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertEqual({r['year'] for r in result['predictions']},{2021})
        data.manifest['records'][0]['record']['year']=2022
        with tempfile.TemporaryDirectory() as folder,self.assertRaises(ValueError):
            compare.make_observer(model,initial,data,[{'indices':[0]}],self.config(),10,'centralized',Path(folder),lambda *a,**k:None)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required for production-loop binding')
    def test_nine_real_training_loops_probes_crash_recovery_and_async_pause_are_exact(self):
        cfg=self.config();stats={'target':{'mean':[0.,0.],'std':[1.,1.]}}
        rows=[{'state':str(i//2),'fips':str(i)} for i in range(8)]
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            root=Path(folder);prepared=root/'prepared';prepared.mkdir();write_json(prepared/'manifest.json',{'fixture':True})
            resumed=root/'resumed';resumed.mkdir();reference=root/'reference';reference.mkdir()
            write_json(resumed/'config.json',cfg)
            control=local.PauseControl(resumed);normal=local.PauseControl(reference)
            calls=0;real_probe=compare.probe_measurement
            def interrupted_probe(*args):
                nonlocal calls
                calls+=1
                if calls==2:raise RuntimeError('simulated interruption after checkpoint and before probe')
                return real_probe(*args)
            with patch.object(local,'CountyData',Data),patch.object(full,'fit_statistics',lambda _:stats), \
                 patch.object(full,'make_model',lambda _:Toy()),patch.object(full,'records',lambda _:rows):
                with patch.object(compare,'probe_measurement',interrupted_probe),self.assertRaises(RuntimeError):
                    compare.run_training(cfg,cfg,prepared,resumed/'training',lambda *a,**k:None,control)
                self.assertEqual(local.read(resumed/'training/centralized_seed10/recovery.json')['event_count'],1)
                def pause_async(stage,**row):
                    if row.get('method')=='hierarchical_async' and row.get('version')==3:control.request()
                with self.assertRaises(local.LocalPause):
                    compare.run_training(cfg,cfg,prepared,resumed/'training',pause_async,control)
                self.assertEqual(local.read(resumed/'training/hierarchical_async_seed10/recovery.json')['event_count'],4)
                (resumed/'PAUSE').unlink()
                def crash_uncommitted(stage,**row):
                    if row.get('method')=='hierarchical_async' and row.get('version')==6:
                        raise RuntimeError('injected uncommitted event failure')
                with self.assertRaisesRegex(RuntimeError,'uncommitted'):
                    compare.run_training(cfg,cfg,prepared,resumed/'training',crash_uncommitted,control)
                compare.run_training(cfg,cfg,prepared,resumed/'training',lambda *a,**k:None,control)
                archives=list((resumed/'curves/hierarchical_async_seed10').glob('uncommitted_before_recovery_*'))
                self.assertEqual(len(archives),1)
                self.assertTrue((archives[0]/'point_00005.json').exists())
                # Independent original loop: no observation wrappers at all.
                local.run_training(cfg,cfg,prepared,reference/'training',lambda *a,**k:None,normal)
            for seed in cfg['seeds']:
                for method in full.METHODS:
                    a=torch.load(resumed/f'training/{method}_seed{seed}/final.pt',weights_only=True)
                    b=torch.load(reference/f'training/{method}_seed{seed}/final.pt',weights_only=True)
                    torch.testing.assert_close(a,b,rtol=0,atol=0)
                    points=plots.committed_points(resumed,method,seed)
                    self.assertEqual([p['event_count'] for p in points],list(range(17 if method=='hierarchical_async' else 5)))
                    self.assertEqual(points[-1]['accepted_epoch_equivalent'],4)
                    if method=='hierarchical_async':self.assertTrue(all(p['ring'] is not None for p in points[1:]))
            self.assertEqual(len(local.read(resumed/'training/training_summary.json')),9)
            self.assertTrue(Path(plots.plot_curves(resumed)).exists())

    def test_live_plot_omits_uncommitted_events_and_statistics_do_not_invent_values(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);method='hierarchical_async';seed=10;arm=root/'training/hierarchical_async_seed10';arm.mkdir(parents=True)
            curves=root/'curves/hierarchical_async_seed10';curves.mkdir(parents=True)
            write_json(arm/'recovery.json',{'event_count':4})
            for i in range(7):
                path=curves/f'point_{i:05d}.json';write_json(path,{'event_count':i,'identity':{'scope':'fixed_training_probe_not_heldout','method':method,'seed':seed}})
                write_json(path.with_suffix('.sha256.json'),{'sha256':sha(path)})
            self.assertEqual([p['event_count'] for p in plots.committed_points(root,method,seed)],list(range(5)))
            self.assertEqual(plots.statistics([1.,2.,3.]),{'mean':2.,'sample_sd':1.,'n':3})
            self.assertIsNone(plots.statistics([1.,None,3.])['mean'])

    def test_final_three_seed_table_uses_per_seed_scores_and_paired_gaps(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg=self.config();results=[]
            for index,method in enumerate(full.METHODS):
                for seed in cfg['seeds']:
                    y=[1.,2.,3.];p=[1.1+index*.1+seed*.001,1.8,3.1]
                    results.append({'method':method,'seed':seed,'metrics':local.evaluate(y,p),'rmse_gap_percent':float(index)+seed/100})
                    write_json(root/f'{method}_seed{seed}_predictions.json',[{'truth':a,'prediction':b} for a,b in zip(y,p)])
            write_json(root/'metrics.json',results);write_json(root/'config.json',cfg)
            write_json(root/'status.json',{'status':'completed','stage':'historical_test','runs':9})
            plots.plot_results(root)
            summary=local.read(root/'seed_summary.json')
            self.assertAlmostEqual(summary['fedavg']['paired_rmse_gap_percent']['mean'],1.2)
            self.assertAlmostEqual(summary['fedavg']['paired_rmse_gap_percent']['sample_sd'],.1)
            report=(root/'comparison.md').read_text(encoding='utf-8')
            self.assertIn('±',report);self.assertIn('balanced',report);self.assertNotIn('nan',report.lower())
            self.assertTrue((root/'paper_table.pdf').exists())
            write_json(root/'metrics.json',results[:-1])
            with self.assertRaises(ValueError):plots.plot_results(root)

    def test_raw_inventory_differentiates_missing_files_from_size_metadata_ready(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'a').write_bytes(b'a');write_json(root/'a.source.json',{'revision':'v','sha256':sha(root/'a')})
            info=compare.raw_inventory({'revision':'v','files':[{'path':'a','size':1},{'path':'b','size':123}]},root)
            self.assertEqual(info['missing_bytes'],123);self.assertEqual(info['size_and_metadata_ready'],1)
            self.assertFalse(info['raw_content_hashes_rechecked_here'])
            (root/'a.source.json').unlink()
            orphan=compare.raw_inventory({'revision':'v','files':[{'path':'a','size':1}]},root)
            self.assertEqual(orphan['size_and_metadata_ready'],0)
            self.assertEqual(orphan['orphan_files_needing_upstream_verification'],['a'])

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required for evaluation binding')
    def test_nine_frozen_evaluations_resume_without_new_inference_and_reject_changed_predictions(self):
        cfg=self.config();stats={'target':{'mean':[0.,0.],'std':[1.,1.]}}
        rows=[{'state':str(i//2),'fips':str(i)} for i in range(8)]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);training=root/'training';training.mkdir();prepared=root/'prepared';prepared.mkdir()
            write_json(prepared/'manifest.json',{'synthetic_qa_only':True});write_json(training/'normalization.json',stats)
            summary=[]
            for seed in cfg['seeds']:
                for method in full.METHODS:
                    path=training/f'{method}_{seed}.pt';torch.manual_seed(seed);torch.save(compare.vector(Toy()),path)
                    summary.append({'method':method,'seed':seed,'checkpoint':path.name,'checkpoint_sha256':sha(path)})
            write_json(training/'training_summary.json',summary);write_json(training/'status.json',{'status':'running'})
            control=local.PauseControl(root);out=root/'evaluation'
            with self.assertRaises(ValueError):compare.evaluate_models(cfg,training,prepared,out,control,lambda *a,**k:None)
            write_json(training/'status.json',{'status':'completed'})
            with patch.object(local,'CountyData',Data),patch.object(local,'make_model',lambda _:Toy()),patch.object(local,'records',lambda *a:rows):
                compare.evaluate_models(cfg,training,prepared,out,control,lambda *a,**k:None)
                before=local.read(out/'metrics.json');self.assertEqual(len(before),9)
                with patch.object(Toy,'forward',side_effect=AssertionError('Saved evaluations must be reused')):
                    compare.evaluate_models(cfg,training,prepared,out,control,lambda *a,**k:None)
                self.assertEqual(before,local.read(out/'metrics.json'))
                path=out/'centralized_seed10_predictions.json';values=local.read(path);values[0]['prediction']+=1;write_json(path,values)
                with self.assertRaisesRegex(ValueError,'Saved predictions changed'):
                    compare.evaluate_models(cfg,training,prepared,out,control,lambda *a,**k:None)


if __name__=='__main__':unittest.main()
