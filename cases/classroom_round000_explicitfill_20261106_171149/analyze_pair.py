"""Decode and compare the prompt-only pair. Never launch inference."""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'round000_shortprompt_seed42_20261106_162713'
ROUND=ROOT.parents[1]/'round_000'
WORK=ROOT.parents[2]
FFMPEG=WORK/'Blender-Classroom-Reconstruction/runtime/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
OUT=ROOT/'analysis'
W,H=960,544
SAMPLES=[0,15,31,46,55,62,68,77,92,108,123]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def decode(path):
    p=subprocess.Popen([str(FFMPEG),'-v','error','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    frames=[]
    while True:
        data=p.stdout.read(W*H*3)
        if not data:break
        assert len(data)==W*H*3
        frames.append(np.frombuffer(data,dtype=np.uint8).reshape(H,W,3).copy())
    err=p.stderr.read().decode('utf-8','replace');assert p.wait()==0 and not err.strip(),err
    assert len(frames)==124
    return frames
def gray_metrics(frames):
    result=[]
    for i,a in enumerate(frames):
        unknown=np.array(Image.open(ROUND/f'renders/known_masks/{i:04d}.png'))==0
        rgb=a.astype(np.float32);mean=rgb.mean(axis=2)
        gray=(np.ptp(rgb,axis=2)<=8)&(mean>=90)&(mean<=155)
        result.append({'frame':i,'unknown_fraction':float(unknown.mean()),
            'gray_fraction_inside_input_unknown':float(gray[unknown].mean()) if unknown.any() else None})
    return result
def compare_video(left,right,path):
    subprocess.run([str(FFMPEG),'-v','error','-y','-i',str(left),'-i',str(right),
      '-filter_complex','[0:v]scale=480:272[a];[1:v]scale=480:272[b];[a][b]hstack=inputs=2[v]',
      '-map','[v]','-an','-c:v','libx264','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],check=True)
def main():
    OUT.mkdir(exist_ok=True)
    assert json.loads((ROOT/'timing.json').read_text())['exit_code']==0
    for name in ['partial_proxy.mp4','reference.png','camera_trajectory.json']:
        assert sha(ROOT/'inputs'/name)==sha(OLD/'inputs'/name)
    assert (ROOT/'outputs/effective_prompt.txt').read_text().strip()==(ROOT/'inputs/prompt.txt').read_text().strip()
    assert 'style_reference_guidance:' not in (ROOT/'outputs/effective_prompt.txt').read_text()
    expected=json.loads((ROOT/'outputs/round000_seed42.run.json').read_text())['sha256']
    assert sha(ROOT/'outputs/round000_seed42.mp4')==expected
    new=decode(ROOT/'outputs/round000_seed42.mp4');old=decode(OLD/'outputs/round000_seed42.mp4')
    for i in SAMPLES:Image.fromarray(new[i]).save(OUT/f'new_{i:04d}.png')
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
    ids=[0,31,55,92,123];tw,th,header=384,218,34
    canvas=Image.new('RGB',(tw*len(ids),(th+header)*3),'#edf0f3');draw=ImageDraw.Draw(canvas)
    for c,i in enumerate(ids):
        groups=[(Image.open(ROUND/f'renders/partial/{i:04d}.png'),'Same input'),(Image.fromarray(old[i]),'Short prompt'),(Image.fromarray(new[i]),'Explicit fill')]
        for r,(im,label) in enumerate(groups):
            x,y=c*tw,r*(th+header);draw.text((x+8,y+7),f'{label} | frame {i}',font=font,fill='#172331')
            canvas.paste(im.resize((tw,th),Image.Resampling.LANCZOS),(x,y+header))
    canvas.save(OUT/'paired_contact.png')
    compare_video(OLD/'outputs/round000_seed42.mp4',ROOT/'outputs/round000_seed42.mp4',OUT/'short_vs_explicit_fill.mp4')
    compare_video(ROOT/'inputs/partial_proxy.mp4',ROOT/'outputs/round000_seed42.mp4',OUT/'input_vs_output.mp4')
    oldmetrics,newmetrics=gray_metrics(old),gray_metrics(new)
    result={'status':'paired_media_ready','same_input_hashes':True,'effective_prompt_matches':True,
        'both_videos_fully_decoded_frames':124,'remote_local_video_hash_match':True,
        'gray_heuristic':'Channel spread <=8 and mean RGB between 90 and 155, evaluated in the commanded input unknown mask. Not a semantic or camera accuracy metric.',
        'old_gray_metrics':oldmetrics,'new_gray_metrics':newmetrics,
        'selected_frame_comparison':[{'frame':i,'old_gray_fraction':oldmetrics[i]['gray_fraction_inside_input_unknown'],
          'new_gray_fraction':newmetrics[i]['gray_fraction_inside_input_unknown']} for i in [31,55,68,92]],
        'scope':'Prompt-only paired seed 42; no reliable causal attribution to an individual phrase, no multi-seed generalization.'}
    (OUT/'paired_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['selected_frame_comparison'],indent=2))
if __name__=='__main__':main()
