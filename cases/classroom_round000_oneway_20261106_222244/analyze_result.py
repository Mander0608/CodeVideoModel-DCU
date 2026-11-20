"""Compare complete clips and approximately matched commanded camera angles."""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'round000_shortprompt_seed42_20261106_162713'
VARIANT=ROOT.parents[1]/'round_000_oneway_20261106_222244'
WORK=ROOT.parents[2]
FFMPEG=WORK/'Blender-Classroom-Reconstruction/runtime/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
OUT=ROOT/'analysis';W,H=960,544
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def decode(path):
    proc=subprocess.Popen([str(FFMPEG),'-v','error','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    frames=[]
    while True:
        data=proc.stdout.read(W*H*3)
        if not data:break
        assert len(data)==W*H*3
        frames.append(np.frombuffer(data,dtype=np.uint8).reshape(H,W,3).copy())
    err=proc.stderr.read().decode('utf-8','replace');assert proc.wait()==0 and not err.strip(),err
    assert len(frames)==124
    return frames
def gray(arr,mask):
    rgb=arr.astype(np.float32);mean=rgb.mean(axis=2)
    g=(np.ptp(rgb,axis=2)<=8)&(mean>=90)&(mean<=155)
    return float(g[mask].mean()) if mask.any() else None
def pair_video(a,b,out):
    subprocess.run([str(FFMPEG),'-v','error','-y','-i',str(a),'-i',str(b),'-filter_complex',
      '[0:v]scale=480:272[a];[1:v]scale=480:272[b];[a][b]hstack=inputs=2[v]',
      '-map','[v]','-an','-c:v','libx264','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],check=True)
def main():
    OUT.mkdir(exist_ok=True)
    assert json.loads((ROOT/'timing.json').read_text())['exit_code']==0
    actual=ROOT/'outputs/round000_seed42.mp4'
    assert sha(actual)==json.loads((ROOT/'outputs/round000_seed42.run.json').read_text())['sha256']
    assert (ROOT/'outputs/effective_prompt.txt').read_text().strip()==(ROOT/'inputs/prompt.txt').read_text().strip()
    new=decode(actual);old=decode(OLD/'outputs/round000_seed42.mp4')
    traj=json.loads((ROOT/'inputs/camera_trajectory.json').read_text())['frames']
    prior=json.loads((OLD/'inputs/camera_trajectory.json').read_text())['frames']
    matches=[]
    for target in [0,-7.5,-15,-22.5,-30]:
        n=min(range(124),key=lambda i:abs(traj[i]['yaw_degrees']-target))
        o=min(range(56),key=lambda i:abs(prior[i]['yaw_degrees']-target))
        nm=np.array(Image.open(VARIANT/f'renders/known_masks/{n:04d}.png'))==0
        om=np.array(Image.open(ROOT.parents[1]/f'round_000/renders/known_masks/{o:04d}.png'))==0
        matches.append({'target_yaw':target,'old_frame':o,'new_frame':n,'old_commanded_yaw':prior[o]['yaw_degrees'],
         'new_commanded_yaw':traj[n]['yaw_degrees'],'old_gray_fraction':gray(old[o],om),'new_gray_fraction':gray(new[n],nm)})
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);tw,th,head=384,218,38
    sheet=Image.new('RGB',(tw*5,(th+head)*3),'#edf0f3');draw=ImageDraw.Draw(sheet)
    for c,m in enumerate(matches):
        n,o=m['new_frame'],m['old_frame']
        items=[(Image.open(VARIANT/f'renders/partial/{n:04d}.png'),f'One-way input f{n}'),
          (Image.fromarray(old[o]),f'Roundtrip output f{o}'),(Image.fromarray(new[n]),f'One-way output f{n}')]
        for r,(im,label) in enumerate(items):
            x,y=c*tw,r*(th+head);draw.text((x+6,y+9),f'{label} | ~{m["target_yaw"]} deg',font=font,fill='#172331')
            sheet.paste(im.resize((tw,th),Image.Resampling.LANCZOS),(x,y+head))
    sheet.save(OUT/'matched_angle_contact.png')
    for i in sorted({0,15,31,46,62,77,92,108,116,123}|{m['new_frame'] for m in matches}):
        Image.fromarray(new[i]).save(OUT/f'new_{i:04d}.png')
    stats=[]
    for i,arr in enumerate(new):
        mask=np.array(Image.open(VARIANT/f'renders/known_masks/{i:04d}.png'))==0
        stats.append({'frame':i,'commanded_yaw':traj[i]['yaw_degrees'],'input_unknown_fraction':float(mask.mean()),'gray_fraction_inside_unknown':gray(arr,mask)})
    pair_video(ROOT/'inputs/partial_proxy.mp4',actual,OUT/'input_vs_output.mp4')
    pair_video(OLD/'outputs/round000_seed42.mp4',actual,OUT/'roundtrip_vs_oneway_full_clips.mp4')
    result={'status':'passed','decoded_frames_per_video':124,'hash_match':True,'effective_prompt_matches':True,
      'matched_commanded_angles':matches,'new_gray_metrics':stats,
      'limitations':['Commanded angles are not measured output camera poses.','Full-clip comparison is synchronized by frame, not angle; use the contact sheet for approximate angle matches.',
      'Color statistic is a gray-placeholder heuristic, not semantic completion accuracy.','Same duration means the one-way turn is slower. Only one seed.']}
    (OUT/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'matched_angles':matches,'final_frame':stats[-1]},indent=2))
if __name__=='__main__':main()
