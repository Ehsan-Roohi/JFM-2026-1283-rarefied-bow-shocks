"""Render the eight established DSMC formation sequences from their numeric high-particle fields."""
from pathlib import Path
import os, json, math, hashlib, subprocess, shutil, sys
import argparse
repository = Path(__file__).resolve().parents[2]
options = argparse.ArgumentParser(description=__doc__)
options.add_argument('--runs', type=Path, default=repository/'data'/'physics'/'video_runs')
options.add_argument('--output', type=Path, default=repository/'reproduced')
args = options.parse_args()
root = args.output.resolve()
base = args.runs.resolve()
root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(root/'.mplconfig'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.patches import Polygon
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter
import cv2
from datetime import datetime,timezone
qa=root/'qa'; out=root/'media'
qa.mkdir(exist_ok=True); out.mkdir(exist_ok=True)
gases=('Ar','N2'); ks=('0p01','0p05','0p1','0p2'); bodies=('cylinder','plate')
ff=shutil.which('ffmpeg')
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':13,
    'axes.edgecolor':'#424c53','axes.labelcolor':'#232d33','xtick.color':'#34414a','ytick.color':'#34414a'})
def npz(g,b,k): return base/'npz_hi'/f'hi_{g}_{b}_Kn{k}.npz'
def smooth(q):
    # Same temporal mean and NaN-aware spatial filter as the original movie renderer.
    q=q.astype('float32')
    avg=q.copy()
    avg[1:-1]=(q[:-2]+q[1:-1]+q[2:])/3
    mask=np.isfinite(avg)
    w=gaussian_filter(mask.astype('float32'),sigma=(0,1,1))
    values=gaussian_filter(np.where(mask,avg,0),sigma=(0,1,1))
    np.divide(values,np.maximum(w,1e-6),out=values);values[~mask]=np.nan
    return values
def screened_stress(raw):
    rho,temp,ratio=raw[:,0],raw[:,1],raw[:,3].astype('float32')
    missing_pressure=(~np.isfinite(rho)) | (~np.isfinite(temp)) | (rho<=0) | (temp<=0)
    impossible=np.isfinite(ratio) & (np.abs(ratio)>3)
    removed=np.isfinite(ratio) & (missing_pressure | impossible)
    ratio[missing_pressure | impossible]=np.nan
    return ratio, {'failed_realizability':int(impossible.sum()),'finite_ratio_at_unresolved_pressure':int((np.isfinite(raw[:,3]) & missing_pressure).sum()),'union_removed':int(removed.sum()),'finite_archive_ratios':int(np.isfinite(raw[:,3]).sum())}
def measured():
    dest=qa/'formation_scales.json'
    if dest.exists() and json.loads(dest.read_text()).get('stress_screen_version')==2: return json.loads(dest.read_text())
    extents={}; screenings={}; maxima=[0.,0.,0.,0.]
    for g in gases:
        for k in ks:
            for b in bodies:
                f=npz(g,b,k)
                with np.load(f,allow_pickle=False) as z: arr=z['fields']; times=z['t']
                assert arr.shape[:2]==(100,4) and np.isclose(times[-1],.0024)
                bounds=[]
                for q in range(4):
                    if q==2 and g=='Ar':bounds.append([0.,0.]);continue
                    if q==3:
                        quantity,screenings[f.name]=screened_stress(arr)
                        values=smooth(quantity)
                    else:values=smooth(arr[:,q])
                    lo=float(np.nanmin(values));hi=float(np.nanmax(values))
                    assert np.isfinite([lo,hi]).all() and (q==3 or lo>=0)
                    bounds.append([lo,hi]);maxima[q]=max(maxima[q],abs(lo),abs(hi))
                extents[f.name]=bounds
    assert maxima[3]<=3.000001
    result={'individual':extents,'stress_screen_version':2,'archived_field_screening':screenings,'density_limit':math.ceil(maxima[0]/5)*5,
            'temperature_limit':math.ceil(max(maxima[1:3])/5)*5,'stress_limit':3,
            'stress_linear_core':3,'display_filter':'interior three-frame average; finite-data Gaussian, sigma one binned grid cell, original mask'}
    dest.write_text(json.dumps(result,indent=2),encoding='utf8');print('SCALES',maxima,result['stress_limit'],flush=True)
    return result
def surf(body):
    text=(base/(body+'.surf')).read_text().splitlines();a=text.index('Points')+1;b=text.index('Lines')
    return np.asarray([[float(x) for x in l.split()[1:3]] for l in text[a:b] if l.strip()])
surfaces={b:surf(b) for b in bodies}
D=float(np.ptp(surfaces['cylinder'][:,1]))
assert np.isclose(D,.3048) and np.isclose(np.ptp(surfaces['plate'][:,1]),D)
def series(g,b,k):
    f=npz(g,b,k)
    with np.load(f,allow_pickle=False) as z:
        m=json.loads(str(z['meta']));ts=z['t'].copy();raw=z['fields']
    assert m['gas']==g and m['body']==b and np.isclose(m['Kn'],float(k.replace('p','.')))
    nbin=math.ceil(m['nx']/480);x0=m['domain'][0]/D;dx=m['dx']*nbin/D;dy=m['dy']*nbin/D
    extent=(x0,x0+raw.shape[-1]*dx,raw.shape[-2]*dy)
    data={i:smooth(screened_stress(raw)[0] if i==3 else raw[:,i]) for i in ([0,1,2,3] if g=='N2' else [0,1,3])}
    return ts,extent,data
def render(n,g,k,sc):
    stem=f'JFM1283_M{n:02}_{g}_Kn{k}_formation';final=out/(stem+'.mp4');cert=qa/(stem+'.json')
    if cert.exists() and final.exists() and json.loads(cert.read_text()).get('stress_screen_version')==2:return
    vals=[series(g,b,k) for b in bodies];ts=vals[0][0]
    assert np.allclose(ts,vals[1][0],rtol=4e-4,atol=1e-9)
    fig=plt.figure(figsize=(12.8,8),dpi=135,facecolor='white')
    grid=fig.add_gridspec(2,2,left=.057,right=.755,top=.865,bottom=.085,wspace=.17,hspace=.265)
    lim=sc['stress_limit'];rl=sc['density_limit'];tl=sc['temperature_limit']
    norms=[SymLogNorm(1, vmin=0,vmax=rl,base=2),Normalize(0,tl),Normalize(0,tl),Normalize(-lim,lim)]
    maps=['turbo','inferno','inferno','RdBu_r']
    artist=[]
    for col,(tsb,(x0,x1,y1),d) in enumerate(vals):
        for idx,array in d.items():
            low=float(np.nanmin(array));high=float(np.nanmax(array))
            assert low>=norms[idx].vmin and high<=norms[idx].vmax,(g,k,idx,low,high)
        for row,(top,bottom) in enumerate(((0,3),(1,2 if g=='N2' else 1))):
            ax=fig.add_subplot(grid[row,col]);ax.set_facecolor('#e3e6e7')
            for q,ext,flipped in ((top,(x0,x1,0,y1),False),(bottom,(x0,x1,-y1,0),True)):
                arr=d[q];fr=arr[0][::-1] if flipped else arr[0]
                im=ax.imshow(fr,aspect='equal',origin='lower',extent=ext,cmap=maps[q],norm=norms[q],interpolation='nearest')
                artist.append((im,arr,flipped))
            ax.axhline(0,color='#eef2f4',lw=.6)
            ax.add_patch(Polygon(surfaces[bodies[col]]/D,facecolor='#454c50',edgecolor='#c7cccf',lw=.5,zorder=4))
            ax.set(xlim=(x0,x1),ylim=(-y1,y1))
            ax.tick_params(labelsize=11,pad=3);ax.locator_params(axis='both',nbins=5)
            if row==0:ax.set_title(('Circular cylinder','Normal flat plate')[col],fontsize=17,pad=8)
            else:ax.set_xlabel(r'$x/D$',labelpad=2)
            if col==0:ax.set_ylabel(r'$y/D$',labelpad=2)
    title=f"Mach 10   ·   {('Argon' if g=='Ar' else 'Nitrogen')}   ·   "+r"$Kn_\infty$"+f" = {k.replace('p','.')}"
    fig.text(.059,.963,title,fontsize=22,va='center')
    clock=fig.text(.966,.963,'',fontsize=19,va='center',ha='right')
    x=.79;w=.175
    def bar(y,index,ticks):
        ax=fig.add_axes([x,y,w,.021])
        cb=fig.colorbar(ScalarMappable(norm=norms[index],cmap=maps[index]),cax=ax,orientation='horizontal',ticks=ticks)
        cb.ax.set_xticklabels([f'{t:g}' for t in ticks]);cb.ax.tick_params(labelsize=11,length=3,pad=3)
    fig.text(x,.79,'Density above centreline',fontsize=15)
    fig.text(x,.75,r'$\rho/\rho_\infty$',fontsize=16)
    bar(.704,0,[v for v in (0,1,2,5,10,rl) if v<=rl])
    fig.text(x,.624,'Normal stress below',fontsize=15)
    fig.text(x,.584,r'$(P_{xx}-P_{yy})/p$',fontsize=16)
    bar(.536,3,sorted(set([-lim,-2,-1,0,1,2,lim])))
    fig.add_artist(plt.Rectangle((x,.444),.011,.015,transform=fig.transFigure,facecolor='#e3e6e7',edgecolor='#80888e',lw=.5))
    fig.text(x+.017,.445,'No resolved value',fontsize=11,color='#424c53')
    fig.text(x,.285,'Temperature',fontsize=16)
    fig.text(x,.242,r'$T_{\rm tr}/T_\infty$'+('  above' if g=='N2' else ''),fontsize=15)
    if g=='N2':fig.text(x,.203,r'$T_{\rm rot}/T_\infty$  below',fontsize=15)
    bar(.148,1,sorted(set([0,10,20,30,tl])))
    fig.canvas.draw()
    width,height=fig.canvas.get_width_height();assert width%2==height%2==0
    tmp=out/(stem+'.partial.mp4')
    cmd=[ff,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgba','-video_size',f'{width}x{height}','-framerate','12','-i','pipe:0','-an','-c:v','libx264','-threads','3','-preset','veryfast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for i,t in enumerate(ts):
            for img,arr,flip in artist: img.set_data(arr[i][::-1] if flip else arr[i])
            clock.set_text(f't = {t*1000:.3f} ms');fig.canvas.draw()
            proc.stdin.write(fig.canvas.buffer_rgba())
        proc.stdin.close()
        if proc.wait():raise RuntimeError('Encoding failed')
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
        plt.close(fig)
    subprocess.run([ff,'-v','error','-xerror','-i',str(tmp),'-f','null','-'],check=True)
    cap=cv2.VideoCapture(str(tmp))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==100 and int(cap.get(cv2.CAP_PROP_FPS))==12
    for index in (0,49,99):
        cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,a=cap.read();assert ok
        if index==49:cv2.imwrite(str(qa/(stem+'.jpg')),cv2.resize(a,(1280,800)))
    cap.release();os.replace(tmp,final)
    with final.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
    cert.write_text(json.dumps({'file':final.name,'bytes':final.stat().st_size,'sha256':sha,'frames':100,'fps':12,
        'dimension':[width,height],'decoded_to_end':True,'stress_range':[-lim,lim],'stress_screen_version':2,'coverage':'every finite displayed value inside colourbar bounds', 'physical_time_ms':[round(ts[0]*1000,4),round(ts[-1]*1000,4)]}),encoding='utf8')
    print('READY',n,final.name,flush=True)
if __name__=='__main__':
    sc=measured()
    for gidx,g in enumerate(gases):
        for kidx,k in enumerate(ks):render(5+4*gidx+kidx,g,k,sc)
