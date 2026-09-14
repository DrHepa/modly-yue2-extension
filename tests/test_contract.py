from __future__ import annotations
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from yue2_modly.common import ROOT, EXTENSION_ID, write_json, read_json, safe_child, sha256
from yue2_modly.config import normalize, manifest, request_dict, native_generation_dict, boolean
from yue2_modly.paths import resolve_models_root
from yue2_modly.installer import choose_lane, git_blob
from yue2_modly.provision import verified_existing
from yue2_modly import bundles


class ManifestTests(unittest.TestCase):
    def test_unique_dispatch_and_types(self):
        data = manifest()
        self.assertEqual(data["type"], "process")
        self.assertTrue((ROOT / data["entry"]).is_file())
        nodes = data["nodes"]
        self.assertEqual(len(nodes), 13)
        self.assertEqual(len({n['id'] for n in nodes}),len(nodes))
        for node in nodes:
            with self.subTest(node=node['id']):
                self.assertIn(node['input'],('audio','text','mesh','image'))
                self.assertIn(node['output'],('audio','text','mesh','image'))
                self.assertEqual(len(node['params_schema']),len({f['id'] for f in node['params_schema']}))
                for field in node['params_schema']:
                    self.assertIn(field['type'],('select','string','int','float'))
                params=normalize(node['id'],{})
                self.assertEqual(params,{f['id']:f['default'] for f in node['params_schema']})

    def test_manifest_regeneration(self):
        expected=(ROOT/'manifest.json').read_bytes()
        subprocess.run([sys.executable,str(ROOT/'tools/build_manifest.py')],check=True,capture_output=True)
        self.assertEqual(expected,(ROOT/'manifest.json').read_bytes())

    def test_upstream_sampling_defaults(self):
        cfg=native_generation_dict(normalize('generate',{}))
        self.assertEqual(cfg['abc'],dict(temperature=.7,top_p=.9,top_k=30,repetition_penalty=1.005,penalty_window=100,min_tokens=32,max_tokens=4096))
        self.assertEqual(cfg['semantic'],dict(temperature=1.,top_p=.95,top_k=100,repetition_penalty=1.2,penalty_window=50,min_tokens=200,max_tokens=9000))
        self.assertEqual(cfg['ode_steps'],32)

    def test_invalid_params(self):
        for params in ({'seed':1.5},{'seed':True},{'seed':-1},{'cfg_scale':float('nan')},
                       {'semantic_min_tokens':500,'semantic_max_tokens':10},
                       {'backend':'vllm','quantization':'fp8'},{'offload_ar':'maybe'},
                       {'decoder':'invented'},{'lyrics':[]}):
            with self.subTest(params=params),self.assertRaises((ValueError,TypeError)):
                normalize('generate',params)

    def test_false_is_false(self):
        self.assertFalse(boolean('false'))
        self.assertFalse(boolean(False))
        self.assertTrue(boolean('true'))

    def test_unknown_node(self):
        with self.assertRaises(ValueError):normalize('does-not-exist',{})

    def test_request_json_validation(self):
        params=normalize('generate',{})
        self.assertIsNone(request_dict(params,lyrics='hola')['cfg_scale'])
        for row in ({'id':'../../x'},{'cot':'off','abc':'K:C\nC D E F'},
                    {'seed':True},{'seed':2**63},{'cfg_scale':float('inf')},{'lyrics':[]},{'shell':'x'}):
            with self.subTest(row=row),self.assertRaises(ValueError):
                request_dict(params,lyrics='',overrides=row)

    def test_lock_revisions_are_full_hashes(self):
        import re
        lock=read_json(ROOT/'upstream.lock.json')
        for sha in [lock['source']['revision'],*lock['source']['git_blobs'].values(),*[x['revision'] for x in lock['weights'].values()]]:
            self.assertRegex(sha,r'^[a-f0-9]{40}$')


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='YuE2 space ñ ')
        self.base=Path(self.tmp.name)
        self.ext=self.base/'host/extensions'/EXTENSION_ID
        self.ext.mkdir(parents=True)
        self.state=self.ext/'.setup-state.json'
        self.models=self.base/'my models'

    def tearDown(self):self.tmp.cleanup()
    def resolve(self,ctx=None,env=None):
        return resolve_models_root(ctx,root=self.ext,state_file=self.state,env={} if env is None else env)

    def test_explicit_models_path(self):
        self.assertEqual(self.resolve({'models_dir':str(self.models)}),self.models)
    def test_host_binding(self):
        write_json(self.base/'host/settings.json',{'extensionsDir':str(self.ext.parent),'modelsDir':str(self.models)})
        self.assertEqual(self.resolve(),self.models)
    def test_unrelated_settings_rejected(self):
        write_json(self.base/'host/settings.json',{'extensionsDir':str(self.base/'unrelated'),'modelsDir':str(self.models)})
        with self.assertRaises(ValueError):self.resolve()
    def test_relocation_overrides_saved_state(self):
        write_json(self.state,{'extension_root':str(self.ext),'models_root':str(self.models)})
        fresh=self.base/'relocated'
        write_json(self.base/'host/settings.json',{'extensionsDir':str(self.ext.parent),'modelsDir':str(fresh)})
        self.assertEqual(self.resolve(),fresh)
    def test_saved_state_bound_to_extension(self):
        write_json(self.state,{'extension_root':str(self.ext),'models_root':str(self.models)})
        self.assertEqual(self.resolve(),self.models)
        write_json(self.state,{'extension_root':str(self.base/'other'),'models_root':str(self.models)})
        with self.assertRaises(ValueError):self.resolve()
    def test_conflicting_paths(self):
        with self.assertRaises(ValueError):self.resolve({'models_dir':str(self.models),'modelsDir':str(self.base/'other')})
    def test_weights_not_inside_extension(self):
        with self.assertRaises(ValueError):self.resolve({'models_dir':str(self.ext/'models')})
    def test_missing_models_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'MODELS_DIR_MISSING'):self.resolve()
        self.assertFalse((self.ext/'models').exists())
    def test_safe_children(self):
        for name in ('../secret','a/../b','/absolute','a\\b','C:foo','','a//b'):
            with self.subTest(name=name),self.assertRaises(ValueError):safe_child(self.base,name)
    def test_symlinks_rejected(self):
        try:(self.base/'link').symlink_to(self.base/'elsewhere')
        except OSError:self.skipTest('symlink privileges unavailable')
        with self.assertRaises(ValueError):safe_child(self.base,'link/file')
    def test_verified_snapshot_reuse_and_corruption(self):
        self.models.mkdir()
        file=self.models/'model.safetensors';file.write_bytes(b'fixture')
        spec={'repo_id':'x/y','revision':'a'*40}
        write_json(self.models/'.complete.json',{**spec,'files':{file.name:{'bytes':file.stat().st_size,'sha256':sha256(file)}}})
        self.assertTrue(verified_existing(self.models,spec))
        file.write_bytes(b'changed')
        self.assertFalse(verified_existing(self.models,spec))
    def test_atomic_utf8_json(self):
        file=self.base/'unicode.json';write_json(file,{'text':'Arañas y canción ♫'})
        self.assertEqual(read_json(file)['text'],'Arañas y canción ♫')
        self.assertEqual(len(list(self.base.glob('*.tmp'))),0)


class LaneTests(unittest.TestCase):
    def identity(self,system='Linux',machine='aarch64',version=(3,11,9)):
        return dict(system=system,machine=machine,version=list(version),implementation='cpython',bits=64)
    def test_six_abi_platform_combinations(self):
        for system,machine in [('Windows','AMD64'),('Linux','x86_64'),('Linux','aarch64')]:
            for minor in (11,12):
                with self.subTest(system=system,machine=machine,minor=minor):
                    lane=choose_lane(self.identity(system,machine,(3,minor,9)),{'cuda_version':128})
                    self.assertEqual(lane['python'],[3,minor,9])
                    self.assertEqual(lane['torch_spec'],'torch==2.10.0+cu128')
    def test_wrong_abi_rejected(self):
        for version in ((3,10,9),(3,13,5)):
            with self.assertRaisesRegex(ValueError,'PYTHON_ABI_UNSUPPORTED'):
                choose_lane(self.identity(version=version),{})
    def test_old_cuda_rejected(self):
        with self.assertRaisesRegex(ValueError,'CUDA_LANE_UNSUPPORTED'):
            choose_lane(self.identity(),{'cuda_version':124})
    def test_windows_arm_is_not_falsely_supported(self):
        with self.assertRaises(ValueError):choose_lane(self.identity('Windows','ARM64'),{})
    def test_git_blob_algorithm(self):
        data=b'hello\n'
        self.assertEqual(git_blob(data),hashlib.sha1(b'blob 6\0hello\n').hexdigest())


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.dir=Path(self.tmp.name)
        np.save(self.dir/'latent.npy',np.zeros((10,64),np.float32),allow_pickle=False)
        self.desc=bundles.seal(self.dir,'latents',{})
    def tearDown(self):self.tmp.cleanup()
    def test_roundtrip(self):
        directory,data=bundles.locate(json.dumps(self.desc))
        self.assertEqual(directory,self.dir)
        self.assertEqual(data['kind'],'latents')
        self.assertEqual(bundles.array(directory/'latent.npy','latent').shape,(10,64))
    def test_modified_data_detected(self):
        (self.dir/'latent.npy').write_bytes(b'bad')
        with self.assertRaises(ValueError):bundles.locate(json.dumps(self.desc))
    def test_modified_manifest_detected(self):
        value=read_json(self.dir/'bundle.json');value['metadata']['changed']=True
        write_json(self.dir/'bundle.json',value)
        with self.assertRaises(ValueError):bundles.locate(json.dumps(self.desc))
    def test_path_traversal_rejected(self):
        value=read_json(self.dir/'bundle.json');value['artifacts']['../escape']={'sha256':'x','bytes':0}
        write_json(self.dir/'bundle.json',value)
        with self.assertRaises(ValueError):bundles.verify(self.dir)
    def test_no_pickle_or_nonfinite(self):
        for value in (np.array([object()],dtype=object),np.full((10,64),np.nan),np.zeros((10,63))):
            np.save(self.dir/'bad.npy',value)
            with self.assertRaises(ValueError):bundles.array(self.dir/'bad.npy','latent')
    def test_semantic_range(self):
        for value in (np.array([-1],dtype=np.int32),np.array([32768]),np.zeros((2,2),np.int32),np.array([1.2])):
            np.save(self.dir/'bad.npy',value)
            with self.assertRaises(ValueError):bundles.array(self.dir/'bad.npy','semantic')


class ProtocolTests(unittest.TestCase):
    def invoke(self,value):
        return subprocess.run([sys.executable,str(ROOT/'processor.py')],input=json.dumps(value)+'\n',text=True,
                              capture_output=True,encoding='utf-8',timeout=30,cwd=ROOT)
    def test_diagnostics_one_terminal(self):
        proc=self.invoke({'input':{'nodeId':'diagnostics'},'params':{}})
        self.assertEqual(proc.returncode,0,proc.stderr)
        lines=[json.loads(x) for x in proc.stdout.splitlines()]
        self.assertEqual([x['type'] for x in lines].count('done'),1)
        self.assertNotIn('error',[x['type'] for x in lines])
        report=json.loads(lines[-1]['result']['text'])
        self.assertFalse(report['inference_executed_by_diagnostics'])
    def test_unknown_node_one_error(self):
        proc=self.invoke({'input':{'nodeId':'unknown'},'params':{}})
        self.assertNotEqual(proc.returncode,0)
        lines=[json.loads(x) for x in proc.stdout.splitlines()]
        self.assertEqual([x['type'] for x in lines].count('error'),1)
        self.assertNotIn('done',[x['type'] for x in lines])
    def test_missing_input_one_error(self):
        proc=self.invoke([])
        lines=[json.loads(x) for x in proc.stdout.splitlines()]
        self.assertEqual(len(lines),1)
        self.assertEqual(lines[0]['type'],'error')
    def test_invalid_parameter_without_loading_model(self):
        proc=self.invoke({'input':{'nodeId':'generate'},'params':{'semantic_top_p':0}})
        self.assertNotEqual(proc.returncode,0)
        self.assertIn('semantic_top_p',json.loads(proc.stdout.splitlines()[-1])['message'])


if __name__=='__main__':unittest.main()
