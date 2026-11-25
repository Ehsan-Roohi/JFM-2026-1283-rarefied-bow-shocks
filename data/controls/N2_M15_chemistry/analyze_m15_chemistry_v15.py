from pathlib import Path
import ast,json,datetime,hashlib,math,csv,tarfile,shutil,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path('campaign');P=Path('campaign')
R=.1524;D=2*R;lambda_nominal=.003048;sgrid=np.linspace(0,.30,3001);ds=sgrid[1]-sgrid[0];PRIMARY_SIGMA=.0015;provenance=[]
source=W/'analyze_nitrogen_parker_and_vibration.py';tree=ast.parse(source.read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['smooth','detect','load_grid','interpolate']],type_ignores=[]),str(source),'exec'))
O=W/('analysis_M15_chemistry_v15_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));O.mkdir()
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
banks={};configs={};statistics={}
for name in ['N2_M15_vibration','N2_M15_chemistry']:
    root=W/'sparta_supplemental_20261022_v1'/name;c=json.loads((root/'config.json').read_text());c['supplemental']=True
    assert c['M']==15 and c['dt']==1e-8 and c['warmup']==200000 and c['block']==20000 and c['target']==400000
    assert (root/'production/TARGET_COMPLETE.json').exists()
    assert 'PRODUCTION_COMPLETE 400000' in (root/'production/log.production').read_text()
    configs[name]=c;profiles={a:[] for a in [170,180]};detail={a:[] for a in profiles}
    for step in range(220000,400001,20000):
        f=root/'production'/f'fields.{step}.grid';grid,h,hy=load_grid(f,c)
        for a in profiles:
            q=np.column_stack([interpolate(grid[k],c,h,hy,a) for k in ['rho','Ttr','u','Trot','lambda','tau','count']]);q[:,0]/=c['nrho']*4.65e-26;profiles[a].append(q)
            marker=detect(q[:,0],max(h,hy));assert marker is not None,(name,step,a)
            detail[a].append(marker)
    banks[name]={a:np.array(v) for a,v in profiles.items()};statistics[name]={}
    for a,bank in banks[name].items():
        q=np.array([r['standoff_m'] for r in detail[a]]);mean=np.nanmean(bank,axis=0);marker=detect(mean[:,0],max(h,hy));assert marker is not None
        statistics[name][str(a)]=dict(mean_profile_marker=marker,block_markers_m=q.tolist(),block_marker_sd_m=float(q.std(ddof=1)),half_drift_percent=float(100*(q[5:].mean()-q[:5].mean())/q.mean()),lag1=float(np.corrcoef(q[:-1],q[1:])[0,1]) if q.std()>0 else None,whole_ray_max_h_lambda=float(np.nanmax(max(h,hy)/bank[:,:,4])),whole_ray_max_dt_tau=float(np.nanmax(c['dt']/bank[:,:,5])))
        np.savez(O/f'{name}_theta{a}.npz',s=sgrid,blocks=bank,mean=mean)
    for f in [root/'config.json',root/'production/in.production',root/'production/TARGET_COMPLETE.json']:provenance.append(dict(file=str(f),sha256=sha(f)))
comparisons=[];mask=(sgrid>=.015)&(sgrid<=.16)
pairs=[('N2_M15_vibration','N2_M15_chemistry','dissociation_on_off_same_Parker_vibration')]
for first,second,label in pairs:
    for angle in [170,180]:
        a=np.nanmean(banks[first][angle],axis=0);b=np.nanmean(banks[second][angle],axis=0)
        sa=statistics[first][str(angle)]['mean_profile_marker']['standoff_m'];sb=statistics[second][str(angle)]['mean_profile_marker']['standoff_m']
        row=dict(comparison=label,first=first,second=second,theta=angle,first_standoff_m=sa,second_standoff_m=sb,standoff_difference_percent=100*(sb-sa)/sa)
        for i,key in enumerate(['rho','Ttr','u','Trot']):
            valid=mask&np.isfinite(a[:,i])&np.isfinite(b[:,i]);row[key+'_L2_percent']=float(100*np.linalg.norm(b[valid,i]-a[valid,i])/np.linalg.norm(a[valid,i]))
        comparisons.append(row)
fig,axes=plt.subplots(1,4,figsize=(18,4))
for name,label in [('N2_M15_vibration','Parker + vibration, dissociation off'),('N2_M15_chemistry','Parker + vibration, dissociation on')]:
    a=np.nanmean(banks[name][170],axis=0)
    for i,ylabel in enumerate(['rho/rhoinf','Ttr (K)','u (m/s)','Trot (K)']):
        axes[i].plot(sgrid[mask],a[mask,i],label=label);axes[i].set(xlabel='Distance outward from wall (m)',ylabel=ylabel);axes[i].legend(fontsize=8)
fig.suptitle('M15 nitrogen; ten equal blocks each, 2–4 ms; rotation and vibration sensitivity');fig.tight_layout();fig.savefig(O/'physical_profiles.png',dpi=160);plt.close(fig)
report=dict(comparison=comparisons,block_statistics=statistics,scientific_convergence=False,window_ms=[2,4],blocks_each=10,limitations='Matched Parker/vibration; mass-density profiles. Check species/counters before interpreting differences as chemistry. Declared seeds are equal but model-dependent stochastic trajectories diverge; no guaranteed noise cancellation. No experimental validation or global mesh convergence. Axis first-row symmetry approximation; whole-ray maxima are not near-wall p95.',configs=configs)
(O/'REPORT.json').write_text(json.dumps(report,indent=2));(O/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2));shutil.copy2(__file__,O/Path(__file__).name);shutil.copy2(source,O/'detector_source.py')

import re
C=W/'sparta_supplemental_20261022_v1';d=C/'N2_M15_chemistry';p=d/'production'
assert 'species '+str(C/'chemistry.species')+' N2 N' in (d/'in.pilot').read_text()
rows=[];inputs=[]
sha=lambda f:hashlib.file_digest(Path(f).open('rb'),'sha256').hexdigest()
for step in range(220000,400001,20000):
 f=p/f'fields.{step}.grid';before=f.stat()
 with f.open() as h:header=[next(h) for _ in range(9)]
 names=header[-1].split()[2:];selected=['c_geom','f_avg[1]','f_densityavg[1]','f_densityavg[2]']
 a=np.loadtxt(f,skiprows=9,usecols=[names.index(k) for k in selected]);assert np.isfinite(a).all()
 volume,count,n2,n=a.T;fluid=volume>0;assert (n2[fluid]>=0).all() and (n[fluid]>=0).all()
 total_bound=2*float(np.sum(n2[fluid]*volume[fluid]));total_free=float(np.sum(n[fluid]*volume[fluid]));total=total_bound+total_free;assert total>0
 local=np.divide(n,2*n2+n,out=np.zeros_like(n),where=(2*n2+n)>0);occupied=fluid&(count>=2)
 rows.append(dict(step=step,time_ms=step*1e-8*1000,free_atomic_fraction_integrated=total_free/total,max_local_free_atomic_fraction_occupied=float(local[occupied].max()),cells_with_nonzero_atomic_density=int(np.sum(fluid&(n>0)))))
 digest=sha(f);after=f.stat();assert(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns);inputs.append(dict(file=str(f),sha256=digest))
log=(p/'log.production').read_text();react=[float(x) for x in re.findall(r'Reactions/particle/step:\s+(\S+)',log)]
r=dict(blocks=rows,input_provenance=inputs,logged_reactions_per_particle_per_step=react,atomic_density_zero_all_sampled_cells=all(row['cells_with_nonzero_atomic_density']==0 for row in rows),limitations='N2,N order verified from species command and builtin species mixture. Fraction counts free N atoms divided by total nitrogen atoms. Sampled2-4ms fields and solver logged reaction counters, not a general chemical-equilibrium or experimental validation. QK dissociation-only; no recombination/ionization/catalysis.')

(O/'CHEMICAL_SPECIES_ADDENDUM.json').write_text(json.dumps(r,indent=2))

dest=P/O.name;dest.mkdir();tar=dest/'analysis.tar';files=list(O.iterdir())
with tarfile.open(tar,'w') as tf:
    for f in files:tf.add(f,arcname=f.name)
with tarfile.open(tar) as tf:
    for f in files:assert hashlib.sha256(tf.extractfile(f.name).read()).hexdigest()==sha(f)
(O/'ARCHIVE_VERIFIED.json').write_text(json.dumps(dict(verified=True,archive=str(tar),sha256=sha(tar))))
print(str(O),flush=True);print(json.dumps(comparisons),flush=True)

