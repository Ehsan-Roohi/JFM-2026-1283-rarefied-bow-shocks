from pathlib import Path
import json,hashlib,ast,datetime,math,numpy as np,tarfile,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path('campaign');P=Path('campaign')
R=.1524;lambda_nominal=.02286;case_lambda=.02286;sgrid=np.linspace(0,.55,5501);ds=.0001;PRIMARY_SIGMA=.0015
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def indices(bounds,angle,s):
    # Intersect a ray with actual leaf-cell rectangles; no structured reshape.
    v=np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))]);origin=np.array([R,0.])+R*v
    low=np.full(len(bounds),-np.inf);high=np.full(len(bounds),np.inf)
    for k in range(2):
        lo=bounds[:,2*k];hi=bounds[:,2*k+1]
        if abs(v[k])<1e-14:
            high=np.where((origin[k]>=lo)&(origin[k]<hi),high,-np.inf)
        else:
            aa=(lo-origin[k])/v[k];bb=(hi-origin[k])/v[k];low=np.maximum(low,np.minimum(aa,bb));high=np.minimum(high,np.maximum(aa,bb))
    valid=np.flatnonzero((high>low+1e-12)&(high>0)&(low<s[-1]));order=np.argsort(low[valid]);ix=valid[order];a=low[ix];b=high[ix]
    assert not np.any(a[1:]<b[:-1]-1e-10),'Overlapping leaf cells on ray'
    j=np.searchsorted(a,s,side='right')-1;safe=np.clip(j,0,len(ix)-1);covered=(j>=0)&(s<=b[safe]+1e-12)
    return np.where(covered,ix[safe],-1)
# A coarse rectangle and two refined adjacent leaves must cover the same ray.
bb=np.array([[0,1,0,1],[1,1.5,0,1],[1.5,2,0,1.]])
# Standalone coverage test uses rayorigin at wall x=0 and points leftward.
ss=np.array([.25,.75,1.25,1.75]);testbounds=bb.copy();testbounds[:,:2]=-bb[:,1::-1]
assert np.array_equal(indices(testbounds,180,ss),[0,0,1,2])
tree=ast.parse((W/'pod_all_kn_v14.py').read_text());functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['smooth','outer']]
exec(compile(ast.Module(body=functions,type_ignores=[]),'validated_outer_detector','exec'))
O=W/('analysis_Kn075_wide_v17_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));O.mkdir()
root=W/'kn075_wide_v12/N2_Kn0075_wide/production';done=json.loads((root/'TARGET_COMPLETE.json').read_text());assert done['blocks']==10
names_needed=['xlo','xhi','ylo','yhi','c_geom','f_avg[1]','f_avg[2]','f_avg[4]','f_avg[8]','c_mfp[1]','c_mfp[2]','f_avg[7]']
banks={str(a):[] for a in [170,180]};geometry=None;provenance=[]
for step in range(110000,200001,10000):
    f=root/f'fields.{step}.grid'
    with f.open() as h:head=[next(h) for _ in range(9)]
    assert int(head[1])==step
    names=head[-1].split()[2:];x=np.loadtxt(f,skiprows=9,usecols=[names.index(n) for n in names_needed]);assert np.isfinite(x).all()
    x=x[np.lexsort((x[:,3],x[:,2],x[:,1],x[:,0]))];bounds=x[:,:4]
    if geometry is None:geometry=bounds.copy()
    else:assert np.array_equal(geometry,bounds)
    for angle in [170,180]:
        ii=indices(bounds,angle,sgrid);valid=ii>=0;safe=np.maximum(ii,0);valid&=x[safe,4]>0
        z=x[safe];h=np.maximum(z[:,1]-z[:,0],z[:,3]-z[:,2]);h=np.where(valid,h,np.nan)
        y=np.column_stack([z[:,6]/5.255e19,z[:,8],z[:,7],np.divide(h,z[:,9],out=np.full(len(h),np.nan),where=z[:,9]>0),np.divide(2e-8,z[:,10],out=np.full(len(h),np.nan),where=z[:,10]>0),z[:,5],z[:,11]])
        y[~valid]=np.nan;banks[str(angle)].append(y)
    provenance.append(dict(step=step,time_ms=step*2e-5,file=str(f),sha256=sha(f)))
report=dict(case='N2_Kn0075_wide',blocks=10,window_ms=[2,4],sampling='Piecewise constant actual cell-ray intersections; solid volume mask; no reshaping/interpolation across cells.',scientific_convergence=False,ray_tests_passed=True,profiles={})
fig,axes=plt.subplots(1,4,figsize=(18,4))
for angle,bank in banks.items():
    bank=np.array(bank);mean=np.nanmean(bank,axis=0);h=float(np.nanmax(np.maximum(geometry[:,1]-geometry[:,0],geometry[:,3]-geometry[:,2])))
    markers=[outer(sgrid,v[:,0],PRIMARY_SIGMA) for v in bank];marker=outer(sgrid,mean[:,0],PRIMARY_SIGMA)
    q=np.array([t['s'] for t in markers if t is not None]);supported=len(q)
    lag=float(np.corrcoef(q[:-1],q[1:])[0,1]) if supported==10 and np.std(q)>0 else None
    r=dict(mean_profile_marker=marker,block_markers_m=[t['s'] if t else None for t in markers],supported_block_markers=supported,block_marker_sd_m=float(np.std(q,ddof=1)) if supported==10 else None,lag1=lag,half_drift_percent=float(100*(q[5:].mean()-q[:5].mean())/q.mean()) if supported==10 else None,max_h_over_lambda=float(np.nanmax(bank[:,:,3])),max_dt_over_tau=float(np.nanmax(bank[:,:,4])),smoothing_sensitivity={str(sig):outer(sgrid,mean[:,0],sig) for sig in [.00075,.0015,.003]},particle_count_ray_p5_p50_p95=np.nanpercentile(bank[:,:,5],[5,50,95]).tolist(),limitations='Unsupported outer compression markers remain None. Nominal lambda .02286 m specific to Kn .075; whole-ray maxima are not nearwall p95; one mesh/run is not convergence. Downstream window limited by actual positive coverage; no near-wall gradient shortcut.')
    report['profiles'][angle]=r
    np.savez(O/f'ray{angle}.npz',s=sgrid,blocks=bank,mean=mean)
    display=(sgrid>=.015)&(sgrid<=.5)
    for i,(col,label) in enumerate([(0,'rho/rhoinf'),(1,'Ttr (K)'),(2,'u (m/s)'),(6,'Trot (K)')]):axes[i].plot(sgrid[display],mean[display,col],label=f'{angle}deg');axes[i].set(xlabel='Distance outward from wall (m)',ylabel=label,xlim=(.015,.5));axes[i].legend()
fig.suptitle('N2 wall L0; actual-cell ray samples; ten blocks, 2–4 ms');fig.tight_layout();fig.savefig(O/'physical_profiles.png',dpi=150);plt.close(fig)
(O/'REPORT.json').write_text(json.dumps(report,indent=2));(O/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2));shutil.copy2(__file__,O/Path(__file__).name)
D=P/O.name;D.mkdir();tar=D/'analysis.tar'
files=list(O.iterdir())
with tarfile.open(tar,'w') as tf:
    for f in files:tf.add(f,arcname=f.name)
with tarfile.open(tar) as tf:
    for f in files:assert hashlib.sha256(tf.extractfile(f.name).read()).hexdigest()==sha(f)
(O/'ARCHIVE_VERIFIED.json').write_text(json.dumps(dict(verified=True,archive=str(tar),sha256=sha(tar))))
print(str(O),flush=True);print(json.dumps(report),flush=True)


