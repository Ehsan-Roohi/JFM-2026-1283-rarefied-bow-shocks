from pathlib import Path
import ast,json,datetime,hashlib,math,csv,tarfile,shutil,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path('campaign');P=Path('campaign')
R=.1524;D=2*R;lambda_nominal=.003048;sgrid=np.linspace(0,.30,3001);ds=sgrid[1]-sgrid[0];PRIMARY_SIGMA=.0015;provenance=[]
source=W/'analyze_nitrogen_parker_and_vibration.py';tree=ast.parse(source.read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['smooth','detect','load_grid','interpolate']],type_ignores=[]),str(source),'exec'))
O=W/('analysis_transport_v24_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));O.mkdir()
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
banks={};configs={};statistics={}
for name in ['N2_seed2','N2_M10_vss']:
    root=W/'sparta_supplemental_20261022_v1'/name;c=json.loads((root/'config.json').read_text());c['supplemental']=True
    assert c['M']==10 and c['dt']==2e-8 and c['warmup']==100000 and c['block']==10000 and c['target']==200000
    assert (root/'production/TARGET_COMPLETE.json').exists()
    assert 'PRODUCTION_COMPLETE 200000' in (root/'production/log.production').read_text()
    configs[name]=c;profiles={a:[] for a in [170,180]};detail={a:[] for a in profiles}
    for step in range(110000,200001,10000):
        f=root/'production'/f'fields.{step}.grid';grid,h,hy=load_grid(f,c)
        for a in profiles:
            q=np.column_stack([interpolate(grid[k],c,h,hy,a) for k in ['nrho','Ttr','u','Trot','lambda','tau','count']]);q[:,0]/=c['nrho'];profiles[a].append(q)
            marker=detect(q[:,0],max(h,hy));assert marker is not None,(name,step,a)
            detail[a].append(marker)
    banks[name]={a:np.array(v) for a,v in profiles.items()};statistics[name]={}
    for a,bank in banks[name].items():
        q=np.array([r['standoff_m'] for r in detail[a]]);mean=np.nanmean(bank,axis=0);marker=detect(mean[:,0],max(h,hy));assert marker is not None
        statistics[name][str(a)]=dict(mean_profile_marker=marker,block_markers_m=q.tolist(),block_marker_sd_m=float(q.std(ddof=1)),half_drift_percent=float(100*(q[5:].mean()-q[:5].mean())/q.mean()),lag1=float(np.corrcoef(q[:-1],q[1:])[0,1]) if q.std()>0 else None,whole_ray_max_h_lambda=float(np.nanmax(max(h,hy)/bank[:,:,4])),whole_ray_max_dt_tau=float(np.nanmax(c['dt']/bank[:,:,5])))
        np.savez(O/f'{name}_theta{a}.npz',s=sgrid,blocks=bank,mean=mean)
    for f in [root/'config.json',root/'production/in.production',root/'production/TARGET_COMPLETE.json']:provenance.append(dict(file=str(f),sha256=sha(f)))
ca,cb=configs['N2_seed2'],configs['N2_M10_vss']
for key in ['nx','ny','dt','ppc','M','nrho','U','xlo','xhi','ly','lz','warmup','block','target','fnum']:assert ca[key]==cb[key],key
comparisons=[];mask=(sgrid>=.015)&(sgrid<=.16)
pairs=[('N2_seed2','N2_M10_vss','VHS_to_first_CE_viscosity_matched_VSS')]
for first,second,label in pairs:
    for angle in [170,180]:
        a=np.nanmean(banks[first][angle],axis=0);b=np.nanmean(banks[second][angle],axis=0)
        sa=statistics[first][str(angle)]['mean_profile_marker']['standoff_m'];sb=statistics[second][str(angle)]['mean_profile_marker']['standoff_m']
        row=dict(comparison=label,first=first,second=second,theta=angle,first_standoff_m=sa,second_standoff_m=sb,standoff_difference_percent=100*(sb-sa)/sa)
        for i,key in enumerate(['rho','Ttr','u','Trot']):
            valid=mask&np.isfinite(a[:,i])&np.isfinite(b[:,i]);row[key+'_L2_percent']=float(100*np.linalg.norm(b[valid,i]-a[valid,i])/np.linalg.norm(a[valid,i]))
        comparisons.append(row)
fig,axes=plt.subplots(1,4,figsize=(18,4))
for name,label in [('N2_seed2','VHS alpha1, constant Z5'),('N2_M10_vss','VSS alpha1.6, matched first CE viscosity, Z5')]:
    a=np.nanmean(banks[name][170],axis=0)
    for i,ylabel in enumerate(['rho/rhoinf','Ttr (K)','u (m/s)','Trot (K)']):
        axes[i].plot(sgrid[mask],a[mask,i],label=label);axes[i].set(xlabel='Distance outward from wall (m)',ylabel=ylabel);axes[i].legend(fontsize=8)
fig.suptitle('M10 nitrogen | VHS/VSS first-CE-viscosity match | ten blocks,2–4ms | 170degree ray');fig.tight_layout();fig.savefig(O/'Supplement_N2_VHS_VSS_transport_matched.png',dpi=160);fig.savefig(O/'Supplement_N2_VHS_VSS_transport_matched.pdf',bbox_inches='tight');plt.close(fig)
report=dict(comparison=comparisons,block_statistics=statistics,scientific_convergence=False,window_ms=[2,4],blocks_each=10,limitations='VHSalpha1 versus VSSalpha1.6,diameter adjusted to match firstChapmanEnskogviscosity at identical omega,Tref; constantZ5,no vibration/chemistry. Independent seeds918173vs438719; no paired noise cancellation or significance claim. Same uniform1552x800grid,dt20ns,ppc20,n,U and window2–4ms, not C4mesh or a finest-grid transport test. Not experimentalvalidation or globalconvergence. Axis isfirstgasrow;whole-raymaxima are not nearwallp95.',configs=configs)
(O/'REPORT.json').write_text(json.dumps(report,indent=2));(O/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2));shutil.copy2(__file__,O/Path(__file__).name);shutil.copy2(source,O/'detector_source.py')
dest=P/O.name;dest.mkdir();tar=dest/'analysis.tar';files=list(O.iterdir())
with tarfile.open(tar,'w') as tf:
    for f in files:tf.add(f,arcname=f.name)
with tarfile.open(tar) as tf:
    for f in files:assert hashlib.sha256(tf.extractfile(f.name).read()).hexdigest()==sha(f)
(O/'ARCHIVE_VERIFIED.json').write_text(json.dumps(dict(verified=True,archive=str(tar),sha256=sha(tar))))
print(str(O),flush=True);print(json.dumps(comparisons),flush=True)


(W/'transport_sensitivity_v24_20261031/RESULT_POINTER.json').write_text(json.dumps({'path':str(O),'archive':str(tar),'sha256':sha(tar),'verified':True},indent=2))
