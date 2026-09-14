"""Opt-in real inference smoke test, run locally AFTER setup; not part of unit CI."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workspace',required=True,type=Path)
parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
parser.add_argument('--backend',choices=('torch','torch-eager','vllm'),default='torch')
parser.add_argument('--memory-budget-gib',type=float,default=24)
parser.add_argument('--max-semantic-tokens',type=int,default=256)
args=parser.parse_args()
workspace=args.workspace.expanduser().resolve();workspace.mkdir(parents=True,exist_ok=True)
payload={'input':{'nodeId':'generate','text':'[Verse]\nSilver notes across the sky\nQuiet stars are passing by\n'},
         'params':{'style':'gentle acoustic folk, clear vocals, guitar','device':args.device,'backend':args.backend,
                   'memory_budget_gib':args.memory_budget_gib,'semantic_min_tokens':0,
                   'semantic_max_tokens':args.max_semantic_tokens,'abc_max_tokens':1024,'abc_min_tokens':0},
         'workspaceDir':str(workspace),'tempDir':str(workspace/'temp')}
proc=subprocess.Popen([sys.executable,str(ROOT/'processor.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                      stderr=None,text=True,encoding='utf-8',cwd=ROOT)
proc.stdin.write(json.dumps(payload)+'\n');proc.stdin.close()
terminal=[]
for line in proc.stdout:
    print(line,end='',flush=True)
    msg=json.loads(line)
    if msg['type'] in ('done','error'):terminal.append(msg)
returncode=proc.wait()
if returncode or len(terminal)!=1 or terminal[0]['type']!='done':
    raise SystemExit('Real inference smoke test FAILED; inspect logs')
file=Path(terminal[0]['result']['filePath'])
import soundfile as sf
import numpy as np
from yue2.storage import verify_result
from yue2_modly.bundles import verify
samples,rate=sf.read(str(file),always_2d=True)
assert rate==48000 and samples.shape[1]==2 and samples.shape[0]>0 and np.isfinite(samples).all()
verify(file.parent);verify_result(file.parent)
print('PASS: real local inference, 48 kHz stereo and artifact hashes. The short token cap can intentionally truncate a song; this is not a musical-quality evaluation.')
