"""Run in the provisioned extension venv; validates native API signatures without weights."""
import inspect
from yue2 import YuE2Pipeline
from yue2.pipeline import SymbolicPlan, SemanticResult, SongResult
from yue2.modeling_vae import YuE2VAE
from yue2.protocol import GenerationConfig

checks = [
    (YuE2Pipeline.from_pretrained, ('local_files_only','vae')),
    (YuE2Pipeline.plan, ('request','cancelled','on_token')),
    (YuE2Pipeline.generate_semantic, ('plan','sampling','cancelled','on_token')),
    (YuE2Pipeline.synthesize, ('semantic','cancelled')),
    (YuE2Pipeline.decode, ('latents','full')),
    (YuE2VAE.encode, ('audio','sample','generator','return_info')),
    (YuE2VAE.decode_tiled, ('latent','core_frames','halo_frames','on_progress')),
]
for func, names in checks:
    signature = inspect.signature(func)
    for name in names:
        if name not in signature.parameters:
            raise RuntimeError(f'Native API drift: {func.__qualname__} has no {name}')
    print(func.__qualname__, signature)
assert GenerationConfig().context == 24576
print('Native API signature check passed. No model was loaded or inferred.')
