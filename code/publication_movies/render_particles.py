"""Replace the small burned-in key, preserving the native plot rectangle, frames and clock."""
from pathlib import Path
import json, os, subprocess, shutil, hashlib
from PIL import Image, ImageDraw, ImageFont
import cv2, numpy as np
import argparse
from matplotlib.font_manager import FontProperties, findfont

repository = Path(__file__).resolve().parents[2]
options = argparse.ArgumentParser(description=__doc__)
options.add_argument('--input', type=Path, required=True, help='Folder containing the four original native-resolution molecule films.')
options.add_argument('--output', type=Path, default=repository/'reproduced')
options.add_argument('--font', type=Path, help='Optional regular TrueType font.')
options.add_argument('--bold-font', type=Path, help='Optional bold TrueType font.')
args = options.parse_args()
root = args.output.resolve(); src = args.input.resolve()
out = root/'media'; out.mkdir(parents=True,exist_ok=True)
qa = root/'qa'; qa.mkdir(exist_ok=True)
def legend_font(path,windows_name,weight,size):
    if path is None:
        font_dir=Path(os.environ.get('WINDIR',''))/'Fonts'
        choice=font_dir/windows_name
        path=choice if choice.is_file() else findfont(FontProperties(family='DejaVu Sans',weight=weight))
    return ImageFont.truetype(str(path),size)
font=legend_font(args.font,'segoeui.ttf','normal',31)
bold=legend_font(args.bold_font,'segoeuib.ttf','bold',30)
im=Image.new('RGB',(1920,66),(10,17,31)); d=ImageDraw.Draw(im); y=32
items=[('title','Particle motion'),('dot','#76baff','Downstream'),('dot','#ffac52','Upstream'),('gap',44),('title','Profiles'),('line','#67bdf4','Density'),('line','#f16669','Translational temperature'),('gap',24),('dots','#d6dce6','Counting line')]
def width(item):
    k=item[0]
    if k=='gap':return item[1]
    if k=='title':return d.textlength(item[1],font=bold)+24
    return 40+d.textlength(item[2],font=font)+26
total=sum(map(width,items));assert total<1860,total
x=int((1920-total)/2)
for item in items:
    k=item[0]
    if k=='title': d.text((x,y),item[1],font=bold,fill='#ffffff',anchor='lm')
    elif k!='gap':
        c,label=item[1:]
        if k=='dot':d.ellipse((x+9,y-8,x+25,y+8),fill=c)
        if k=='line':d.line((x,y,x+32,y),fill=c,width=5)
        if k=='dots':
            for xx in (x,x+10,x+20,x+30):d.ellipse((xx,y-2,xx+3,y+1),fill=c)
        d.text((x+40,y),label,font=font,fill='#e8edf5',anchor='lm')
    x+=width(item)
leg=qa/'particle-key.png';im.save(leg)
ff=shutil.which('ffmpeg')
for i,(g,k) in enumerate((('Ar','0p01'),('Ar','0p1'),('N2','0p01'),('N2','0p1')),1):
    original=src/f'molecules_{g}_Kn{k}.mp4'
    stem=f'JFM1283_M{i:02}_{g}_Kn{k}_particles'
    final=out/(stem+'.mp4'); cert=qa/(stem+'.json')
    if cert.exists() and final.exists():
        s=json.loads(cert.read_text());assert s['bytes']==final.stat().st_size;continue
    tmp=out/(stem+'.partial.mp4')
    args=[ff,'-nostdin','-hide_banner','-loglevel','error','-y','-i',str(original),'-i',str(leg),'-filter_complex','[0:v][1:v]overlay=0:1014:eof_action=repeat:format=yuv420[v]','-map','[v]','-map','0:a?','-c:v','libx264','-threads','4','-preset','veryfast','-crf','17','-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart',str(tmp)]
    subprocess.run(args,check=True)
    a=cv2.VideoCapture(str(original)); b=cv2.VideoCapture(str(tmp))
    for field in (cv2.CAP_PROP_FRAME_COUNT,cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FPS):
        assert a.get(field)==b.get(field),(field,a.get(field),b.get(field))
    assert b.get(cv2.CAP_PROP_FRAME_COUNT)==900 and b.get(cv2.CAP_PROP_FPS)==30
    diffs=[]
    for fr in (0,450,899):
        a.set(cv2.CAP_PROP_POS_FRAMES,fr); b.set(cv2.CAP_PROP_POS_FRAMES,fr)
        ok1,ai=a.read();ok2,bi=b.read();assert ok1 and ok2
        diff=cv2.absdiff(ai[:1014],bi[:1014]).mean();diffs.append(round(float(diff),3))
        assert diff<6 and cv2.absdiff(ai[1018:],bi[1018:]).mean()>2
        if fr==450:cv2.imwrite(str(qa/(stem+'.jpg')),cv2.resize(bi,(1280,720)))
    a.release();b.release()
    subprocess.run([ff,'-v','error','-xerror','-i',str(tmp),'-f','null','-'],check=True)
    os.replace(tmp,final)
    with final.open('rb') as fh: digest=hashlib.file_digest(fh,'sha256').hexdigest()
    cert.write_text(json.dumps({'file':final.name,'sha256':digest,'bytes':final.stat().st_size,'frames':900,'fps':30,'width':1920,'height':1080,'decoded_to_end':True,'original_plot_mean_abs_rgb_difference':diffs,'legend_pixel_height':66}),encoding='utf-8')
    print('READY',i,final.name,final.stat().st_size,flush=True)
