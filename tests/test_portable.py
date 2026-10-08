"""Standalone checkout guards, independent of local historical result folders."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from research.joint_fusion import prepare_mmst_compare as prep
from research.joint_fusion.mmst_io import write_json


class PortableTests(unittest.TestCase):
    def test_complete_pinned_manifest(self):
        cfg=prep.compare.cal.read(prep.compare.CONFIG)
        manifest=prep.compare.cal.read(prep.compare.ROOT/'evidence/download_manifest.json')
        prep.validate_manifest(cfg,manifest)
        with self.assertRaises(ValueError):
            prep.validate_manifest(cfg,{**manifest,'files':manifest['files'][:-1]})

    def test_fresh_prepare_and_resume_do_not_require_historical_runs(self):
        cfg=prep.compare.cal.read(prep.compare.CONFIG)
        manifest=prep.compare.cal.read(prep.compare.ROOT/'evidence/download_manifest.json')
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)/'fresh';base.mkdir()
            control=prep.compare.local.PauseControl(base)
            calls=[]
            def prepared(config,raw,out,evaluation,progress):
                calls.append('prepared_train')
                out.mkdir(parents=True)
                write_json(out/'status.json',{'status':'completed'})
            with patch.object(prep.compare,'fetch_resilient') as fetch, \
                 patch.object(prep.compare.local,'prepare',prepared), \
                 patch.object(prep.compare,'ensure_inputs',return_value=(base/'test',{'status':'completed'})):
                prep.prepare_inputs(cfg,manifest,base,control,lambda *a,**k:None)
                prep.prepare_inputs(cfg,manifest,base,control,lambda *a,**k:None)
                self.assertEqual(calls,['prepared_train'])
                self.assertEqual(fetch.call_count,2)
                self.assertEqual(prep.compare.cal.read(base/'config.json'),cfg)
                prep.compare.local.full.check_sources(base)
                write_json(base/'config.json',{**cfg,'epochs':119})
                with self.assertRaises(ValueError):
                    prep.prepare_inputs(cfg,manifest,base,control,lambda *a,**k:None)


if __name__ == '__main__': unittest.main()
