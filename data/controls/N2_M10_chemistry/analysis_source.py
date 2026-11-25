"""Snapshot analysis of completed DSMC sampling blocks; never opens live checkpoints.

Regular-grid bilinear sampling avoids triangulation across the solid. Density
markers follow the *outer* compression identified from an upstream plateau.
All numerical definitions and unresolved limitations are saved with results.
"""
from pathlib import Path
import json,datetime,time,hashlib,subprocess,math,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W=Path('campaign')
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_chemistry'
O=W/('analysis_snapshot_'+stamp);O.mkdir()
R=.1524;D=2*R;lambda_nominal=.01*D
angles=np.linspace(100,180,65);sgrid=np.linspace(0,.30,3001);ds=sgrid[1]-sgrid[0]
PRIMARY_SIGMA=.0015
history=[];metrics=[];resolution=[];inventory=[];provenance=[];banks={};fullmaps={};skips=[]

def smooth(y,sigma):
    n=math.ceil(4*sigma/ds);z=np.arange(-n,n+1)*ds;k=np.exp(-.5*(z/sigma)**2);k/=k.sum()
    return np.convolve(np.pad(y,n,mode='reflect'),k,mode='valid')

def detect(profile,h,threshold=.2,span=10,sigma=PRIMARY_SIGMA,downwin=(4,8)):
    finite=np.flatnonzero(np.isfinite(profile))
    if len(finite)<80:return None
    lo,hi=finite[0],finite[-1]
    if not np.isfinite(profile[lo:hi+1]).all():return None
    ss=sgrid[lo:hi+1];raw=profile[lo:hi+1];q=smooth(raw,sigma);g=-np.gradient(q,ds)
    upmask=(ss>ss[-1]-.035)&(ss<ss[-1]-.005)
    if upmask.sum()<20:return None
    up=float(np.median(q[upmask]));updev=abs(up-1)
    if updev>.10:return None
    above=q>up*(1+threshold)
    starts=np.flatnonzero(np.diff(np.r_[False,above,False].astype(int))==1)
    ends=np.flatnonzero(np.diff(np.r_[False,above,False].astype(int))==-1)-1
    runs=[(a,b) for a,b in zip(starts,ends) if (b-a)*ds>=max(3*h,.003)]
    if not runs:return None
    a,b=runs[-1];onset=ss[b]
    bracket=(ss>onset-span*lambda_nominal)&(ss<onset+2*lambda_nominal)&(ss>2*h)
    ii=np.flatnonzero(bracket)
    if len(ii)<20:return None
    im=ii[np.argmax(g[ii])]
    if im<=ii[0]+1 or im>=ii[-1]-1 or g[im]<=0:return None
    peak=float(ss[im])
    dw=(ss>peak-downwin[1]*lambda_nominal)&(ss<peak-downwin[0]*lambda_nominal)
    uw=(ss>peak+4*lambda_nominal)&(ss<peak+8*lambda_nominal)
    if dw.sum()<10 or uw.sum()<10:return None
    down=float(np.median(q[dw]));localup=float(np.median(q[uw]));jump=down-localup
    if jump<.5:return None
    delta=float(jump/g[im])
    level=(down+localup)/2
    crossings=np.flatnonzero((q[:-1]-level)*(q[1:]-level)<=0)
    crossings=[i for i in crossings if abs(ss[i]-peak)<8*lambda_nominal]
    half=np.nan
    if crossings:
        i=min(crossings,key=lambda i:abs(ss[i]-peak))
        if q[i+1]!=q[i]:half=float(ss[i]+ds*(level-q[i])/(q[i+1]-q[i]))
    unrestricted=np.flatnonzero((ss>.015*R)&(ss<.94*ss[-1]))
    old=float(ss[unrestricted[np.argmax(abs(g[unrestricted]))]]) if len(unrestricted) else np.nan
    return dict(standoff_m=peak,delta_m=delta,half_jump_m=half,onset_m=float(onset),
        max_gradient_rhoinf_per_m=float(g[im]),rho_up=localup,rho_down=down,
        upstream_boundary_median=up,upstream_boundary_deviation=updev,
        downstream_window_rel_spread=float(np.std(q[dw])/max(abs(down),1e-30)),
        unrestricted_absolute_marker_m=old,cells_across_delta=delta/h)

def load_grid(f,c):
    stat=f.stat()
    with f.open() as handle:header=[next(handle) for _ in range(9)]
    names=header[-1].split()[2:]
    supplemental=c.get('supplemental',False)
    temp=14 if supplemental else (7 if c['gas']=='Ar' else 9)
    selected=['xc','yc','c_geom','f_avg[1]','f_avg[2]','f_avg[3]','f_avg[4]',f'f_avg[{temp}]','c_mfp[1]','c_mfp[2]']
    if c['gas']=='N2':selected+=['f_avg[7]']
    a=np.loadtxt(f,skiprows=9,usecols=[names.index(k) for k in selected],ndmin=2)
    assert np.isfinite(a).all() and len(a)==c['nx']*c['ny']
    h=(c['xhi']-c['xlo'])/c['nx'];hy=c['ly']/c['ny']
    ix=np.rint((a[:,0]-c['xlo'])/h-.5).astype(int);iy=np.rint(a[:,1]/hy-.5).astype(int)
    assert len(np.unique(iy*c['nx']+ix))==len(a)
    arrays={}
    aliases={'c_geom':'volume','f_avg[1]':'count','f_avg[2]':'nrho','f_avg[3]':'rho','f_avg[4]':'u',f'f_avg[{temp}]':'Ttr','c_mfp[1]':'lambda','c_mfp[2]':'tau'}
    if c['gas']=='N2':aliases['f_avg[7]']='Trot'
    for col,key in aliases.items():
        ar=np.full((c['ny'],c['nx']),np.nan);ar[iy,ix]=a[:,selected.index(col)]
        if key!='volume':ar[arrays['volume']<=0]=np.nan
        arrays[key]=ar
    with f.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
    after=f.stat();assert (stat.st_size,stat.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Field changed while reading'
    provenance.append(dict(file=str(f),size=stat.st_size,sha256=digest,mtime_utc=datetime.datetime.fromtimestamp(stat.st_mtime,datetime.timezone.utc).isoformat()))
    return arrays,h,hy

def interpolate(a,c,h,hy,theta):
    t=np.deg2rad(theta);x=R+(R+sgrid)*np.cos(t);y=(R+sgrid)*np.sin(t)
    xx=(x-c['xlo'])/h-.5;yy=y/hy-.5
    # Even reflection of scalars at the prescribed y=0 symmetry boundary.
    # For 0<=y<=hy/2 this equals the first cell-centre row.
    yy=np.maximum(yy,0)
    valid=(xx>=0)&(xx<=c['nx']-1)&(yy<=c['ny']-1)
    xx=np.clip(xx,0,c['nx']-1);yy=np.clip(yy,0,c['ny']-1)
    i=np.minimum(np.floor(xx).astype(int),c['nx']-2);j=np.minimum(np.floor(yy).astype(int),c['ny']-2)
    wx=xx-i;wy=yy-j
    v=(1-wy)*((1-wx)*a[j,i]+wx*a[j,i+1])+wy*((1-wx)*a[j+1,i]+wx*a[j+1,i+1])
    v[~valid]=np.nan;return v

def savecsv(filename,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (O/filename).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

cases=[];job_records=[]
for name,gas in [('sparta_Ar_Kn001_20261022_v1','Ar'),('sparta_N2_Kn001_20261022_v1','N2')]:
    base=W/name
    for r in json.loads((base/'PRODUCTION_SUBMISSIONS.json').read_text()):
        c=json.loads((Path(r['root']).parent/'config.json').read_text())
        c.update(gas=gas,geometry='cylinder',xlo=-.2,xhi=.65,ly=.4,lz=1.,nrho=3.856e20 if gas=='Ar' else 3.941e20,production_root=r['root'],job=r['job'])
        factor=round(2e-8/c['dt']);c.update(warmup=100000*factor,block=10000*factor,target=200000*factor)
        cases.append(c);job_records.append(dict(case=r['case'],job=r['job']))
sup=W/'sparta_supplemental_20261022_v1'
if sup.exists():
    rec=json.loads((sup/'SUBMISSIONS.json').read_text());cfg=json.loads((sup/'CAMPAIGN.json').read_text())
    for c in cfg['configs']:
        r=next(x for x in rec if x['case']==c['name'] and x['stage']=='production')
        job_records.append(dict(case=c['name'],job=r['job']))
        if c['geometry']=='cylinder':
            c.update(supplemental=True,production_root=str(Path(c['root'])/'production'),job=r['job']);cases.append(c)

cases=[c for c in cases if c['name']=='N2_M10_chemistry']
status=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(r['job'] for r in job_records),'-o','JobIDRaw,State,ExitCode,Elapsed'],text=True)
(O/'SCHEDULER_SNAPSHOT.txt').write_text(status)
states={row.split('|')[0]:row.split('|')[1:] for row in status.splitlines() if row.strip()}
for r in job_records:r['state'],r['exit'],r['elapsed']=states.get(r['job'],['UNKNOWN','',''])
savecsv('all_production_jobs.csv',job_records)
for c in cases:
    p=Path(c['production_root']);completed=states.get(c['job'],[''])[0]=='COMPLETED'
    if completed:assert states[c['job']][1]=='0:0'
    selected=[]
    for f in sorted(p.glob('fields.*.grid'),key=lambda f:int(f.name.split('.')[1])):
        step=int(f.name.split('.')[1])
        # A dump at warmup is created immediately upon defining the fix, before
        # a full sampling block exists, and must never enter the analysis.
        if step<=c['warmup']:continue
        if step>c['target']:continue
        if not completed and time.time()-f.stat().st_mtime<120:continue
        selected.append((step,f))
    inv=dict(case=c['name'],gas=c['gas'],job=c['job'],state=states.get(c['job'],[''])[0],target_complete=completed,
        sampled_blocks=len(selected),steps=[n for n,_ in selected],dt=c['dt'],nx=c['nx'],ny=c['ny'],ppc=c['ppc'])
    inventory.append(inv)
    if not selected:continue
    if completed:assert len(selected)==10 and selected[-1][0]==c['target']
    mass=6.63e-26 if c['gas']=='Ar' else 4.65e-26;rhoinf=c['nrho']*mass
    bank=dict(config=c,complete=completed,blocks=[],profiles=[])
    for step,f in selected:
        print('READ',c['name'],step,flush=True)
        try:a,h,hy=load_grid(f,c)
        except Exception as err:
            if completed:raise
            skips.append(dict(file=str(f),error=str(err)));continue
        prof={key:np.array([interpolate(a[key],c,h,hy,theta) for theta in angles]) for key in ['rho','Ttr','u']+(['Trot'] if c['gas']=='N2' else [])}
        prof['rho']/=rhoinf
        blockmetrics=[]
        for it,theta in enumerate(angles):
            peak=detect(prof['rho'][it],max(h,hy))
            if peak is None:continue
            row=dict(case=c['name'],gas=c['gas'],step=step,time_ms=step*c['dt']*1000,theta=theta,completed=completed,**peak)
            metrics.append(row);blockmetrics.append(row)
        bank['blocks'].append(step);bank['profiles'].append(prof)
        if c['name']=='Ar_grid776':fullmaps.setdefault(c['name'],[]).append(a['rho']/rhoinf)
        axis=next((r for r in blockmetrics if r['theta']==180),None)
        if axis:
            xc=c['xlo']+(np.arange(c['nx'])+.5)*h;yc=(np.arange(c['ny'])+.5)*hy
            xx,yy=np.meshgrid(xc,yc);ss=np.sqrt((xx-R)**2+yy**2)-R;th=np.rad2deg(np.arctan2(yy,xx-R))
            valid=(a['volume']>0)&(a['count']>=2)&(ss>=0)&np.isfinite(a['lambda'])&(a['lambda']>0)&(a['tau']>0)
            shock=(xx>-axis['standoff_m']-2*axis['delta_m'])&(xx<-axis['standoff_m']+2*axis['delta_m'])&(yy<.02)
            wall=(ss<.02*R)&(th>=140)
            for region,mask in [('near_axis_shock',shock),('front_wall_0p02R',wall)]:
                mask &= valid
                if not mask.any():continue
                hl=max(h,hy)/a['lambda'][mask];tt=c['dt']/a['tau'][mask]
                resolution.append(dict(case=c['name'],step=step,region=region,cells=int(mask.sum()),h_m=max(h,hy),
                    h_over_lambda_median=float(np.median(hl)),h_over_lambda_p95=float(np.quantile(hl,.95)),h_over_lambda_max=float(hl.max()),
                    dt_over_tau_p95=float(np.quantile(tt,.95)),dt_over_tau_max=float(tt.max())))
        # Extraction sensitivity uses the block profile, without fitting to a
        # continuum standoff or to the desired mesh-convergence result.
        for theta in [170.,180.]:
            q=prof['rho'][np.flatnonzero(angles==theta)[0]]
            for sigma in [.001,.0015,.002,.003]:
                result=detect(q,max(h,hy),sigma=sigma)
                if result:history.append(dict(case=c['name'],step=step,theta=theta,test='physical_smoothing',setting=sigma,**result))
            for threshold in [.1,.2,.3]:
                result=detect(q,max(h,hy),threshold=threshold)
                if result:history.append(dict(case=c['name'],step=step,theta=theta,test='onset_threshold',setting=threshold,**result))
            for span in [6,10,14]:
                result=detect(q,max(h,hy),span=span)
                if result:history.append(dict(case=c['name'],step=step,theta=theta,test='search_span_lambda',setting=span,**result))
            for win in [(3,6),(4,8),(5,10)]:
                result=detect(q,max(h,hy),downwin=win)
                if result:history.append(dict(case=c['name'],step=step,theta=theta,test='downstream_window_lambda',setting=str(win),**result))
        del a
    if bank['blocks']:banks[c['name']]=bank

savecsv('case_inventory.csv',inventory);savecsv('ray_block_metrics.csv',metrics);savecsv('extraction_sensitivity.csv',history);savecsv('local_resolution.csv',resolution)
savecsv('skipped_live_files.csv',skips)
summary=[]
for name,bank in banks.items():
    for theta in [170.,180.]:
        rows=[r for r in metrics if r['case']==name and r['theta']==theta]
        if not rows:continue
        out=dict(case=name,gas=bank['config']['gas'],theta=theta,completed=bank['complete'],n_blocks=len(rows),time_start_ms=bank['config']['warmup']*bank['config']['dt']*1000,time_end_ms=rows[-1]['time_ms'])
        for key in ['standoff_m','delta_m','half_jump_m','unrestricted_absolute_marker_m','rho_up','rho_down','cells_across_delta']:
            v=np.array([r[key] for r in rows]);out[key+'_mean']=float(np.mean(v));out[key+'_sd']=float(np.std(v,ddof=1)) if len(v)>1 else None
        v=np.array([r['standoff_m'] for r in rows]);n=len(v)
        out['standoff_firsthalf_lasthalf_change_pct']=float(100*(np.mean(v[n//2:])-np.mean(v[:n//2]))/np.mean(v)) if n>=4 else None
        out['standoff_lag1']=float(np.corrcoef(v[:-1],v[1:])[0,1]) if n>=4 and np.std(v[:-1])>0 and np.std(v[1:])>0 else None
        out['note']='Non-overlapping blocks; independence not established. Axis uses even symmetry/first-centre-row approximation.'
        summary.append(out)
savecsv('primary_summary.csv',summary)


reference=W/'analysis_snapshot_20261023T052906Z'
comparison=[];sensitivity=[]
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for theta in [170.,180.]:
    b=banks['N2_M10_chemistry'];assert b['complete'] and len(b['blocks'])==10
    it=np.flatnonzero(angles==theta)[0]
    q=np.column_stack([sgrid]+[np.mean([p[k][it] for p in b['profiles']],axis=0) for k in ['rho','Ttr','u','Trot']])
    np.savetxt(O/f'profile_N2_M10_chemistry_theta{int(theta)}.csv',q,delimiter=',',header='s_m,rho_over_input_rhoinf,Ttr_K,u_m_s,Trot_K',comments='')
    f=reference/f'profile_N2_grid1552_theta{int(theta)}.csv'
    a=np.loadtxt(f,delimiter=',',skiprows=1);assert np.allclose(a[:,0],q[:,0])
    provenance.append(dict(file=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    mask=(sgrid>=.02)&(sgrid<=.12)&np.isfinite(a).all(axis=1)&np.isfinite(q).all(axis=1)
    da=detect(a[:,1],max(.85/1552,.4/800));db=detect(q[:,1],max(.85/1552,.4/800));assert da and db
    r=dict(theta=theta,blocks_per_case=10,physical_window_ms=[2,4],VHS_standoff_m=da['standoff_m'],chemistry_standoff_m=db['standoff_m'],standoff_difference_pct=100*(db['standoff_m']-da['standoff_m'])/da['standoff_m'])
    for j,k in enumerate(['rho','Ttr','u','Trot'],1):r[k+'_L2_pct']=100*float(np.linalg.norm(q[mask,j]-a[mask,j])/np.linalg.norm(a[mask,j]))
    comparison.append(r)
    for model,profile in [('baseline_Zrot5',a),('combined_dissociation',q)]:
        for sigma in [.001,.0015,.002,.003]:
            for threshold in [.1,.2,.3]:
                d=detect(profile[:,1],max(.85/1552,.4/800),sigma=sigma,threshold=threshold)
                sensitivity.append(dict(model=model,theta=theta,sigma_m=sigma,threshold=threshold,standoff_m=d['standoff_m'] if d else None))
    if theta==170:
        for ax,j,label in [(axes[0],1,'rho/rhoinf'),(axes[1],2,'Ttr (K)'),(axes[2],4,'Trot (K)')]:
            ax.plot(sgrid[mask],a[mask,j],label='Constant Zrot5, no vibration');ax.plot(sgrid[mask],q[mask,j],ls='--',label='Parker + vibration + QK dissociation');ax.set(xlabel='Distance outward from wall (m)',ylabel=label);ax.legend();ax.grid(alpha=.2)
fig.suptitle('N2 1552x800; ten blocks each, 2-4 ms; theta=170 deg; different random seeds')
fig.savefig(O/'CHEMISTRY_COMBINED_physical_profiles.png',dpi=180);plt.close(fig)
report=dict(comparison=comparison,chemistry_block_summary=summary,marker_sensitivity=sensitivity,scientific_convergence=False,limitations='Baseline comparison changes rotation, vibration and chemistry together. Additional paired comparison holds Parker and discrete vibration fixed and switches dissociation on/off. QK dissociation-only exploratory model: no recombination, ionization or surface catalysis; not full real-gas validation. Different random seeds, axis approximation and unresolved wall resolution remain.',reference=str(reference))

vibration_comparisons=[]
for theta in [170,180]:
    pf=O/f'profile_N2_M10_chemistry_theta{theta}.csv'
    vf=W/'analysis_snapshot_20261023T133052Z'/f'profile_N2_M10_vibration_theta{theta}.csv'
    a=np.loadtxt(vf,delimiter=',',skiprows=1);b=np.loadtxt(pf,delimiter=',',skiprows=1);assert np.allclose(a[:,0],b[:,0])
    mask=(sgrid>=.02)&(sgrid<=.12)&np.isfinite(a).all(axis=1)&np.isfinite(b).all(axis=1)
    pa=detect(a[:,1],max(.85/1552,.4/800));pb=detect(b[:,1],max(.85/1552,.4/800));assert pa and pb
    r=dict(theta=theta,matched_window_ms=[2,4],blocks_each=10,Parker_vibration_no_dissociation_standoff_m=pa['standoff_m'],Parker_vibration_with_dissociation_standoff_m=pb['standoff_m'],standoff_difference_pct=100*(pb['standoff_m']-pa['standoff_m'])/pa['standoff_m'])
    for j,k in enumerate(['rho','Ttr','u','Trot'],1):r[k+'_L2_pct']=100*float(np.linalg.norm(b[mask,j]-a[mask,j])/np.linalg.norm(a[mask,j]))
    vibration_comparisons.append(r);provenance.append(dict(file=str(vf),sha256=hashlib.sha256(vf.read_bytes()).hexdigest()))
    if theta==170:
        fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
        for ax,j,label in [(axes[0],1,'rho/rhoinf'),(axes[1],2,'Ttr (K)'),(axes[2],4,'Trot (K)')]:
            ax.plot(sgrid[mask],a[mask,j],label='Parker + vibration, no dissociation');ax.plot(sgrid[mask],b[mask,j],ls='--',label='Parker + vibration + dissociation');ax.set(xlabel='Distance outward from wall (m)',ylabel=label);ax.grid(alpha=.2);ax.legend()
        fig.suptitle('Same Parker rotation and vibration; ten blocks each, 2-4 ms; independent random seeds')
        fig.savefig(O/'Parker_vibration_dissociation_on_off.png',dpi=180);plt.close(fig)
report['dissociation_on_off_same_Parker_and_vibration']=vibration_comparisons

(O/'NITROGEN_DISSOCIATION_REPORT.json').write_text(json.dumps(report,indent=2))
(O/'INPUT_PROVENANCE.json').write_text(json.dumps(provenance,indent=2))
import tarfile,shutil,re
c=banks['N2_M10_chemistry']['config'];root=Path(c['root']);inp=root/'production/in.production'
shutil.copy2(root/'config.json',O/'case_config.json');shutil.copy2(inp,O/'source_in.production')
shutil.copy2(root/'production/ARCHIVE_VERIFIED.json',O/'RAW_RESULTS_ARCHIVE_VERIFIED.json')
for i,path in enumerate(re.findall(r'(?:collide vss gas|read_species) (\S+)',inp.read_text())):
    if Path(path).is_file():shutil.copy2(path,O/f'model_{i}_{Path(path).name}')
shutil.copy2(__file__,O/'analysis_source.py')
P=Path('campaign')/O.name;P.mkdir(exist_ok=True)
tarpath=O/'nitrogen_dissociation_analysis.tar';files=list(O.iterdir())
with tarfile.open(tarpath,'w') as t:
    for f in files:t.add(f,arcname=f.name)
dest=P/tarpath.name;shutil.copy2(tarpath,dest)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(tarpath)==sha(dest)
with tarfile.open(dest) as t:
    for f in files:assert hashlib.sha256(t.extractfile(f.name).read()).hexdigest()==sha(f)
receipt=dict(archive=str(dest),sha256=sha(dest),members=len(files),verified=True)
(O/'ARCHIVE_VERIFIED.json').write_text(json.dumps(receipt,indent=2));(P/'ARCHIVE_VERIFIED.json').write_text(json.dumps(receipt,indent=2))
print('ANALYSIS_OUTPUT',O,flush=True);print(json.dumps(report),flush=True)
