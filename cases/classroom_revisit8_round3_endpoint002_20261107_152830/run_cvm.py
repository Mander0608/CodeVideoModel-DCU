"""Portable launch adapter. Default is a dry run; --execute is required for inference.

Example on the existing server:
python run_cvm.py --project-root /public/home/lixin1/chenxy/CodeVideoModel
Use the existing DCU environment/launcher when actually executing this script.
"""
from pathlib import Path
import argparse, hashlib, inspect, json, runpy, sys

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--model-root',type=Path)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(); cfg=json.loads((ROOT/'cvm_config.json').read_text())
    project=args.project_root.resolve(); model=args.model_root or project/'models/MiniMax-H3-ref2va'
    for relative in cfg['inputs'].values(): assert (ROOT/relative).is_file(),relative
    argv=[str(project/'run_dcu.py'),'--code-video',str(ROOT/cfg['inputs']['code_video']),
       '--reference-image',str(ROOT/cfg['inputs']['reference_image']),
       '--prompt-file',str(ROOT/cfg['inputs']['prompt_file']), '--model-root',str(model),
       '--text-device',cfg['text_device'],'--generation-device',cfg['generation_device'],'--keep-steps',str(cfg['keep_steps']),'--seed',str(cfg['seed']),'--output',str(ROOT/cfg['output'])]
    print(json.dumps({'execute':args.execute,'argv':argv,'automatic_style_guidance':False},indent=2))
    if not args.execute: return
    assert (project/'run_dcu.py').is_file(); assert model.is_dir()
    sys.path.insert(0,str(project))
    from cvm import video
    assert video.STEPS==cfg['steps'] and video.FRAMES==cfg['frames']
    source=inspect.getsource(video.generate)
    old='prompt = prompt.rstrip() + "\\n\\nstyle_reference_guidance: " + STYLE_REFERENCE'
    assert source.count(old)==1, 'CVM changed: review guidance adapter before running'
    patched=source.replace(old,'prompt = prompt.rstrip()')
    assert 'STYLE_REFERENCE' not in patched
    out=ROOT/'outputs'; out.mkdir(exist_ok=True)
    (out/'generate_effective.py').write_text(patched,encoding='utf-8')
    prompt=(ROOT/cfg['inputs']['prompt_file']).read_text(encoding='utf-8').strip()
    (out/'effective_prompt.txt').write_text(prompt,encoding='utf-8')
    (out/'launch_record.json').write_text(json.dumps({'argv':argv,'config':cfg,
       'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
       'video_py_sha256':hashlib.sha256((project/'cvm/video.py').read_bytes()).hexdigest()},indent=2),encoding='utf-8')
    exec(compile(patched,str(out/'generate_effective.py'),'exec'),video.__dict__)
    sys.argv=argv
    runpy.run_path(str(project/'run_dcu.py'),run_name='__main__')

if __name__=='__main__': main()
