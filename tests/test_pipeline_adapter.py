"""Integration tests with explicit test doubles, never a production fallback."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
import torch
from yue2_modly import bundles, runtime
from yue2_modly.common import read_json, write_json, sha256, canonical_hash
from yue2_modly.config import normalize


@dataclass
class Request:
    style:str
    lyrics:str
    cot:str='full'
    seed:int=831001
    id:str='song'
    abc:str|None=None
    cfg_scale:float|None=None
    def to_dict(self):return asdict(self)

@dataclass
class Plan:
    request:Request
    abc:str|None
    abc_ids:list
    prefix:list
    timing:dict
    truncated:bool=False
    def save(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        if self.abc is not None:(directory/'score.abc').write_text(self.abc,encoding='utf-8')
        np.save(directory/'abc_tokens.npy',np.asarray(self.abc_ids,dtype=np.int32),allow_pickle=False)
        np.save(directory/'prefix.npy',np.asarray(self.prefix,dtype=np.int32),allow_pickle=False)
        write_json(directory/'plan.json',{'request':self.request.to_dict(),'abc':self.abc,'abc_ids':self.abc_ids,
                                         'prefix':self.prefix,'timing':self.timing,'truncated':self.truncated})
        names=['plan.json','abc_tokens.npy','prefix.npy']+(['score.abc'] if self.abc is not None else [])
        write_json(directory/'plan_manifest.json',{n:sha256(directory/n) for n in names})
    @classmethod
    def load(cls,directory):
        for n,h in read_json(directory/'plan_manifest.json').items():
            if sha256(directory/n)!=h:raise ValueError('Changed exact plan')
        data=read_json(directory/'plan.json')
        return cls(Request(**data['request']),data['abc'],data['abc_ids'],data['prefix'],data['timing'],data['truncated'])

@dataclass
class Semantic:
    plan:Plan
    tokens:list
    timing:dict
    truncated:bool

class Song:
    def __init__(self,audio,sample_rate,semantic,latents,config,weights,timing,request_identity):
        self.audio=audio;self.sample_rate=sample_rate;self.semantic=semantic;self.latents=latents
        self.config=config;self.weights=weights;self.timing=timing;self.request_identity=request_identity
    def save(self,path):
        sf.write(str(path),self.audio,self.sample_rate,subtype='PCM_24' if Path(path).suffix=='.flac' else 'FLOAT')
    def save_artifacts(self,directory):
        self.semantic.plan.save(directory)
        self.save(directory/'audio.flac')
        np.save(directory/'semantic.npy',np.asarray(self.semantic.tokens,dtype=np.int32),allow_pickle=False)
        np.save(directory/'latent.npy',self.latents.astype(np.float32),allow_pickle=False)
        write_json(directory/'request.json',self.semantic.plan.request.to_dict())
        write_json(directory/'config.json',self.config)
        files={p.relative_to(directory).as_posix():{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in directory.rglob('*') if p.is_file() and p.name!='result.json'}
        write_json(directory/'result.json',dict(status='complete',artifacts=files,weights=self.weights,timing=self.timing,
                   identity=self.request_identity,truncated={'abc':False,'semantic':False}))


def verify_native(directory):
    data=read_json(directory/'result.json')
    for name,meta in data['artifacts'].items():
        if sha256(directory/name)!=meta['sha256']:
            raise ValueError('Native result hash mismatch: '+name)
    return data


class Pipe:
    def __init__(self):
        self.weights={'mot':{'files':{'model.safetensors':{'sha256':'m','bytes':1}}},'vae':{'files':{}}}
        self.load_timing={};self.calls=[];self.closed=False
    def __enter__(self):return self
    def __exit__(self,*a):self.closed=True
    def effective_config(self,request):return {'native_fixture':True,'vae_decode':'halo_crop'}
    def plan(self,*,request,cancelled=None,on_token=None):
        self.calls.append(('plan',request))
        if cancelled and cancelled():raise InterruptedError('cancelled')
        abc=None if request.cot=='off' else request.abc or 'X:1\nM:4/4\nL:1/4\nK:C\nC D E F|'
        return Plan(request,abc,[] if abc is None else [31,32],[151643,31,32,151851],{},False)
    def generate_semantic(self,plan,*,cancelled=None,on_token=None):
        self.calls.append(('semantic',plan.prefix))
        if on_token:on_token('semantic',5)
        return Semantic(plan,[5,6,7],{},False)
    def synthesize(self,semantic,*,cancelled=None):
        self.calls.append(('synthesize',semantic.tokens))
        return np.zeros((3,64),np.float32)
    def decode(self,latents,*,full=False):
        self.calls.append(('decode',full))
        return np.zeros((1920*3-64,2),np.float32)


class FakeVAE:
    def to(self,*a,**k):return self
    def encode(self,audio,*,sample=False,generator=None,return_info=False):
        value=torch.zeros((1,64,2),dtype=torch.float32)
        return (value,{'mean':value,'scale':value,'stdev':value}) if return_info else value
    def decode(self,z):return torch.zeros((1,2,1920*z.shape[-1]-64),dtype=torch.float32)
    def decode_tiled(self,z,*,core_frames=None,halo_frames=None,output_device=None,on_progress=None):
        if on_progress:on_progress(1,1)
        return self.decode(z)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='YuE2 ü workspace ')
        self.workspace=Path(self.tmp.name)
        self.pipe=Pipe();self.messages=[]
        pipeline=types.ModuleType('yue2.pipeline');pipeline.SymbolicPlan=Plan;pipeline.SemanticResult=Semantic;pipeline.SongResult=Song
        protocol=types.ModuleType('yue2.protocol');protocol.SongRequest=Request
        storage=types.ModuleType('yue2.storage');storage.identity=canonical_hash;storage.verify_result=verify_native
        self.modules=patch.dict(sys.modules,{'yue2':types.ModuleType('yue2'),'yue2.pipeline':pipeline,'yue2.protocol':protocol,'yue2.storage':storage})
        self.modules.start()
        self.factory=patch('yue2_modly.runtime.make_pipeline',side_effect=lambda *_:self.pipe);self.factory_mock=self.factory.start()
        self.vae=patch('yue2_modly.runtime.vae_model',return_value=(FakeVAE(),{}));self.vae.start()
    def tearDown(self):
        self.vae.stop();self.factory.stop();self.modules.stop();self.tmp.cleanup()
    def run_node(self,node,text='',params=None,file=None):
        payload={'input':{'nodeId':node,'text':text},'params':params or {},'workspaceDir':str(self.workspace),'tempDir':str(self.workspace/'temp')}
        if file:payload['input']['filePath']=str(file)
        return runtime.execute(payload,self.messages.append)
    def directory(self,result):return Path(json.loads(result['text'])['directory'])
    def test_generate_creates_native_audio_and_valid_hashes(self):
        result=self.run_node('generate','Original fixture lyrics')
        audio=Path(result['filePath']);self.assertTrue(audio.is_file())
        self.assertEqual(sf.info(str(audio)).samplerate,48000)
        self.assertEqual(sf.info(str(audio)).channels,2)
        self.assertEqual(bundles.verify(audio.parent)['kind'],'song')
        native=verify_native(audio.parent)
        self.assertNotIn('bundle.json',native['artifacts'])
        self.assertTrue(self.pipe.closed)
    def test_full_staged_roundtrip(self):
        plan=self.run_node('plan','Lyrics')
        sem=self.run_node('semantic',plan['text'])
        latent=self.run_node('synthesize',sem['text'])
        audio=self.run_node('decode',latent['text'],{'device':'cpu'})
        self.assertTrue(Path(audio['filePath']).is_file())
        self.assertEqual(bundles.array(self.directory(sem)/'semantic.npy','semantic').tolist(),[5,6,7])
        self.assertEqual(read_json(self.directory(latent)/'plan.json')['prefix'],[151643,31,32,151851])
    def test_render_exact_saved_plan_does_not_replan(self):
        plan=self.run_node('plan','Lyrics')
        count=sum(x[0]=='plan' for x in self.pipe.calls)
        self.run_node('render-plan',plan['text'])
        self.assertEqual(sum(x[0]=='plan' for x in self.pipe.calls),count)
    def test_render_external_abc(self):
        abc='X:1\nK:C\nC D E F|'
        self.run_node('render-abc',abc,{'lyrics':'Separate lyrics','style':'jazz'})
        request=self.pipe.calls[0][1]
        self.assertEqual(request.abc,abc);self.assertEqual(request.lyrics,'Separate lyrics')
    def test_off_has_no_abc(self):
        result=self.run_node('generate','Lyrics',{'cot':'off'})
        self.assertFalse((Path(result['filePath']).parent/'score.abc').exists())
    def test_request_json_agent_edits_are_new_requests(self):
        data={'style':'rock','lyrics':'Edited lyrics','abc':'X:1\nK:C\nCDEF|','id':'edit-1'}
        self.run_node('generate',json.dumps(data),{'input_mode':'request-json'})
        self.assertEqual(self.pipe.calls[0][1].id,'edit-1')
        self.assertEqual(self.pipe.calls[0][1].abc,data['abc'])
    def test_full_vae_override_recorded(self):
        result=self.run_node('generate','Lyrics',{'full_decode':'true'})
        self.assertEqual(read_json(Path(result['filePath']).parent/'config.json')['vae_decode'],'full')
        self.assertIn(('decode',True),self.pipe.calls)
    def test_inspect_audio_score_and_agent_package(self):
        result=self.run_node('generate','Lyrics')
        audio=Path(result['filePath'])
        score=self.run_node('inspect-audio',params={'extract':'abc'},file=audio)
        self.assertIn('K:C',score['text'])
        descriptor=self.run_node('inspect-audio',params={'extract':'descriptor'},file=audio)
        agent=self.run_node('inspect-bundle',descriptor['text'],{'extract':'agent-package'})
        self.assertIn('not waveform inpainting',json.loads(agent['text'])['instruction'])
    def test_encode_preserves_posterior_and_decode(self):
        file=self.workspace/'source.wav';sf.write(str(file),np.zeros((4800,2),np.float32),48000)
        latent=self.run_node('encode',params={'device':'cpu','save_posterior_info':'true'},file=file)
        directory=self.directory(latent)
        self.assertTrue((directory/'posterior_mean.npy').is_file())
        result=self.run_node('decode',latent['text'],{'device':'cpu'})
        self.assertTrue(Path(result['filePath']).is_file())
    def test_encode_does_not_silently_resample(self):
        file=self.workspace/'bad.wav';sf.write(str(file),np.zeros((4800,1),np.float32),44100)
        with self.assertRaisesRegex(ValueError,'48 kHz stereo'):
            self.run_node('encode',file=file)
    def test_import_native_complete_result(self):
        result=self.run_node('generate','Lyrics')
        source=Path(result['filePath']).parent
        (source/'bundle.json').unlink()
        imported=self.run_node('import-artifacts',str(source))
        verify_native(self.directory(imported))
        self.assertEqual(bundles.verify(self.directory(imported))['kind'],'song')
    def test_import_standalone_native_latents(self):
        directory=self.workspace/'native';directory.mkdir()
        np.save(directory/'latent.npy',np.zeros((2,64),np.float32),allow_pickle=False)
        result=self.run_node('import-artifacts',str(directory))
        self.assertEqual(bundles.verify(self.directory(result))['kind'],'latents')
    def test_batch_resume_skips_only_verified_complete_songs(self):
        rows=[{'style':'jazz','lyrics':'A','id':'a'},{'style':'rock','lyrics':'B','id':'b'}]
        result=self.run_node('batch',json.dumps(rows))
        data=json.loads(result['text']);self.assertEqual(len(data['items']),2)
        count=len(self.pipe.calls)
        resumed=self.run_node('batch',json.dumps(rows),{'resume_directory':data['batch_directory']})
        self.assertEqual(len(self.pipe.calls),count)
        self.assertEqual(len(json.loads(resumed['text'])['items']),2)
        with self.assertRaisesRegex(ValueError,'identity changed'):
            self.run_node('batch',json.dumps(rows),{'resume_directory':data['batch_directory'],'seed':2})
    def test_batch_duplicate_ids_fail_before_pipeline(self):
        row={'style':'x','lyrics':'A','id':'same'}
        with self.assertRaisesRegex(ValueError,'unique'):self.run_node('batch',json.dumps([row,row]))
        self.assertEqual(self.factory_mock.call_count,0)
    def test_changed_model_identity_rejected(self):
        plan=self.run_node('plan','Lyrics')
        self.pipe.weights={'mot':{'changed':True},'vae':{}}
        with self.assertRaisesRegex(ValueError,'different MoT weights'):
            self.run_node('render-plan',plan['text'])
    def test_progress_monotonic(self):
        self.run_node('generate','Lyrics')
        values=[x['percent'] for x in self.messages if x['type']=='progress']
        self.assertEqual(values,sorted(values));self.assertEqual(values[-1],100)
    def test_cancellation_before_loading(self):
        payload={'input':{'nodeId':'generate','text':'Lyrics'},'params':{},'workspaceDir':str(self.workspace)}
        with self.assertRaises(InterruptedError):runtime.execute(payload,self.messages.append,lambda:True)
        self.assertEqual(self.factory_mock.call_count,0)


if __name__=='__main__':unittest.main()
