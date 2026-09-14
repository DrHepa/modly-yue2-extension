"""Explicit native save_pretrained export. This deliberately copies weights on request."""
import argparse
from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from yue2_modly.paths import resolve_models_root, asset_path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('destination',type=Path,help='Empty export directory outside the active checkpoints')
parser.add_argument('--models-root',type=str)
parser.add_argument('--decoder',choices=('default','legacy'),default='default')
args=parser.parse_args()
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from yue2 import YuE2Pipeline
models=resolve_models_root({'models_dir':args.models_root} if args.models_root else {})
destination=args.destination.expanduser().resolve()
if destination.exists() and any(destination.iterdir()):
    raise SystemExit('Export destination must be empty')
if destination.is_relative_to(models):
    raise SystemExit('Export outside active models_dir to avoid accidental checkpoint changes')
with YuE2Pipeline.from_pretrained(str(asset_path(models,'mot')),vae=str(asset_path(models,'vae_legacy' if args.decoder=='legacy' else 'vae')),
                                local_files_only=True,device='cpu',backend='torch-eager',progress=True) as pipe:
    pipe.save_pretrained(destination)
print(destination)
