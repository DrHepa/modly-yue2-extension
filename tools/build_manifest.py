"""Regenerate the shipped manifest using only the audited v0.4.2 parameter types."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def s(id, label, default="", **kwargs):
    return dict(id=id, label=label, type="string", default=default, **kwargs)
def n(id, label, default, low, high, kind="int"):
    return dict(id=id, label=label, type=kind, default=default, min=low, max=high)
def select(id, label, default, values, **kwargs):
    return dict(id=id,label=label,type="select",default=default,options=[dict(value=x,label=x) for x in values],**kwargs)
def b(id, label, default=False):
    return select(id,label,str(default).lower(),["false","true"])

runtime = [select("device","Device","cuda",["cuda","cpu"]),
 select("backend","AR backend","torch",["torch","torch-eager","vllm"]),
 select("quantization","AR quantization (FP8 experimental)","none",["none","fp8"]),
 n("memory_budget_gib","GPU memory budget (GiB)",24,3,512,"float"),
 b("offload_ar","Offload AR during acoustic synthesis"),
 select("decoder","VAE decoder","default",["default","legacy"]),
 n("vae_core_frames","VAE core frames (0 = native default)",0,0,24576),
 b("full_decode","Full VAE decode instead of tiled"),
 select("audio_format","Primary audio format","wav",["wav","flac"])]
request = [s("song_id","Song identifier","song"),s("style","Style tags","cinematic, warm vocals, piano"),
 s("lyrics","Lyrics (text input takes priority)"),s("lyrics_file","Lyrics file",pickerIntent="text"),
 select("cot","Symbolic planning","full",["full","melody","off"]),
 n("seed","Seed",831001,0,2147483647),
 select("cfg_mode","CFG","native",["native","custom"]),
 n("cfg_scale","Custom CFG scale",1.0,0,20,"float")]
sampling=[]
for phase, defaults in (("abc",(.7,.9,30,1.005,100,32,4096)),("semantic",(1.,.95,100,1.2,50,200,9000))):
    for name,value,low,high,kind in zip(
        ("temperature","top_p","top_k","repetition_penalty","penalty_window","min_tokens","max_tokens"),
        defaults,(0,.000001,1,.000001,1,0,1),(5,1,184704,100,100,24576,24576),
        ("float","float","int","float","int","int","int")):
        sampling.append(n(phase+"_"+name,phase.upper()+" "+name.replace("_"," "),value,low,high,kind))
sampling.append(n("ode_steps","Acoustic midpoint ODE steps",32,1,1000))
bundle_input=[s("bundle_path","Bundle / native artifact directory",pickerIntent="folder")]
inspect_params=[select("extract","Text to return","descriptor",["descriptor","abc","request","metadata","agent-package","verify"]),*bundle_input]

def node(id,name,input,output,params,**kwargs):
    return dict(id=id,name=name,input=input,output=output,params_schema=params,**kwargs)
nodes=[
 node("generate","YuE2 · Generate Song","text","audio",[select("input_mode","Text input contains","lyrics",["lyrics","request-json"]),*request,*runtime,*sampling]),
 node("plan","YuE2 · Plan Score","text","text",[select("input_mode","Text input contains","lyrics",["lyrics","request-json"]),*request,*runtime,*sampling]),
 node("render-abc","YuE2 · Render ABC / Cover / Edit","text","audio",[s("abc","External ABC score"),s("abc_file","ABC file",pickerIntent="text"),*request,*runtime,*sampling]),
 node("render-plan","YuE2 · Render Exact Saved Plan","text","audio",[*bundle_input,*runtime,*sampling]),
 node("semantic","YuE2 · Plan to Semantic Tokens","text","text",[*bundle_input,*runtime,*sampling]),
 node("synthesize","YuE2 · Semantic Tokens to Latents","text","text",[*bundle_input,*runtime,*sampling]),
 node("decode","YuE2 · Decode Audio Latents","text","audio",[*bundle_input,*runtime]),
 node("encode","YuE2 · Encode Audio (VAE, not transcription)","audio","text",[
     s("audio_path","48 kHz stereo WAV/FLAC path"),
     select("device","Device","cuda",["cuda","cpu"]),select("decoder","VAE encoder","default",["default","legacy"]),
     b("posterior_sample","Sample posterior (off = posterior mean)"),b("save_posterior_info","Save mean, scale and stdev"),
     n("seed","Posterior seed",831001,0,2147483647)]),
 node("inspect-audio","YuE2 · Audio to Score / Artifacts","audio","text",inspect_params),
 node("inspect-bundle","YuE2 · Inspect Bundle / Agent Package","text","text",inspect_params),
 node("import-artifacts","YuE2 · Import Native Saved Artifacts","text","text",bundle_input),
 node("batch","YuE2 · Batch Requests / Resume","text","text",[
     s("requests_file","Requests JSON / JSONL",pickerIntent="text"),
     s("resume_directory","Existing YuE2 batch directory (optional)",pickerIntent="folder"),
     *request,*runtime,*sampling]),
 node("diagnostics","YuE2 · Installation Diagnostics","text","text",[]),
]
# Do not show controls that cannot affect a particular native stage.
request_ids={f["id"] for f in request}
abc_ids={f["id"] for f in sampling if f["id"].startswith("abc_")}
semantic_ids={f["id"] for f in sampling if f["id"].startswith("semantic_")}
limits={
    "plan": request_ids | abc_ids | {"input_mode","device","backend","quantization","memory_budget_gib"},
    "semantic": semantic_ids | {"bundle_path","device","backend","quantization","memory_budget_gib"},
    "synthesize": {"bundle_path","device","memory_budget_gib","offload_ar","ode_steps"},
    "render-plan": {f["id"] for f in runtime} | semantic_ids | {"bundle_path","ode_steps"},
    "decode": {"bundle_path","device","decoder","vae_core_frames","full_decode","audio_format"},
}
for item in nodes:
    if item["id"] in limits:
        item["params_schema"]=[f for f in item["params_schema"] if f["id"] in limits[item["id"]]]
manifest=dict(id="modly-yue2-extension",name="YuE2",type="process",entry="processor.py",version="0.1.0",
              author="DrHepa",description="YuE2 song generation, symbolic plans, ABC covers and edits, staged audio synthesis and VAE tools. Local setup-managed shared weights.",nodes=nodes)
(ROOT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
print(f"Wrote {len(nodes)} process nodes")
