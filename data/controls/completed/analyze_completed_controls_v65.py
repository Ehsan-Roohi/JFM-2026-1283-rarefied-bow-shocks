from pathlib import Path
import json,hashlib,tarfile,io,datetime,warnings,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore',message='Mean of empty slice')
W=Path('campaign');O=W/'completed_controls_review_v65_20261102';O.mkdir(exist_ok=True)
sources={};archives={}
for tag,folder in [('Ar_grid','argon_grid_review_v58_20261101'),('Ar_dt','argon_dt_review_v62_20261102'),('N2_grid','nitrogen_grid_review_v64_20261102')]:
 r=json.loads((W/folder/'ARCHIVE_VERIFIED.json').read_text());a=Path(r['archive']);assert hashlib.sha256(a.read_bytes()).hexdigest()==r['sha256']
 with tarfile.open(a) as t:
  mf='MANIFEST.json' if 'MANIFEST.json' in t.getnames() else 'MEMBER_SHA256.json'
  m=json.load(t.extractfile(mf));assert set(t.getnames())==set(m)|{mf}
  for n,h in m.items():assert hashlib.sha256(t.extractfile(n).read()).hexdigest()==h,n
  sources[tag]={n:t.extractfile(n).read() for n in m}
 archives[tag]=dict(archive=str(a),sha256=r['sha256'],members=len(m)+1,all_members_verified=True)
 (O/(tag+'_SOURCE_ASSESSMENT.json')).write_bytes(sources[tag]['ASSESSMENT.json'])
banks={};reports={};checks=[]
for tag,levels in [('Ar_grid',['L0','L4','L5']),('N2_grid',['L0','L4','L5']),('Ar_dt',['dt2p5ns'])]:
 for level in levels:
  key=('C4_Ar_'+level if tag=='Ar_grid' else ('C4_N2_'+level+('_40' if level=='L5' else '') if tag=='N2_grid' else 'DT_Ar_L5'))
  report=json.loads(sources[tag][level+'_REPORT.json']);assert report['blocks_each']==10 and report['window_ms']==[2,4]
  c=report['cases'][key];case=tag+'_'+level;reports[case]=c
  provenance=json.loads(sources[tag][level+'_PROVENANCE.json']);fields=[x for x in provenance if 'step' in x]
  assert len(fields)==10 and np.allclose([x['time_ms'] for x in fields],np.arange(2.2,4.01,.2))
  for angle in ['170','180']:
   z=np.load(io.BytesIO(sources[tag][level+'_'+key+'_ray'+angle+'.npz']),allow_pickle=False)
   s=z['s'];b=z['blocks'];mean=z['mean'];assert b.shape==(10,3001,7)
   assert np.allclose(np.nanmean(b,axis=0),mean,equal_nan=True)
   banks[case,angle]=(s,b,mean)
  checks.append(dict(case=case,blocks=10,time_interval_ms=[2,4],end_times_ms=[x['time_ms'] for x in fields],mean_recomputed_from_blocks=True))
# Same Ar L5 bank used for grid and dt studies; verify duplicated source representation.
for angle in ['170','180']:
 z=np.load(io.BytesIO(sources['Ar_dt']['dt5ns_C4_Ar_L5_ray'+angle+'.npz']),allow_pickle=False)
 assert np.array_equal(z['s'],banks['Ar_grid_L5',angle][0]) and np.allclose(z['blocks'],banks['Ar_grid_L5',angle][1],equal_nan=True)
stats=[]
for (case,angle),(s,b,mean) in banks.items():
 for col,field in enumerate(['rho_over_rhoinf','Ttr_K','ux_ms','Trot_K']):
  if field=='Trot_K' and case.startswith('Ar_'):continue
  good=(s>=.015)&(s<=.16)&np.all(np.isfinite(b[:,:,col]),axis=0)&np.isfinite(mean[:,col])
  values=b[:,good,col];mu=mean[good,col];den=np.linalg.norm(mu);assert den>0 and good.sum()>1000
  variability=100*np.sqrt(np.mean(np.sum((values-mu)**2,axis=1)))/den
  drift=100*np.linalg.norm(values[:5].mean(axis=0)-values[5:].mean(axis=0))/den
  stats.append(dict(case=case,ray=int(angle),field=field,window_m=[.015,.16],finite_locations=int(good.sum()),between_block_RMS_percent=float(variability),first5_vs_last5_L2_percent=float(drift),normalization='L2 norm of all-ten-block mean on same common finite physical samples',is_CI=False,is_independent_replicate_estimate=False))
comparisons=[]
for name,base,ref in [('Ar_grid_L4_L5','Ar_grid_L4','Ar_grid_L5'),('Ar_dt_5_2p5ns','Ar_grid_L5','Ar_dt_dt2p5ns'),('N2_grid_L4_L5','N2_grid_L4','N2_grid_L5')]:
 for angle in ['170','180']:
  s,_,a=banks[base,angle];s2,_,b=banks[ref,angle];assert np.array_equal(s,s2)
  r=dict(comparison=name,base=base,reference=ref,ray=int(angle),window_m=[.015,.16],fields={})
  for col,field in enumerate(['rho_over_rhoinf','Ttr_K','ux_ms','Trot_K']):
   if field=='Trot_K' and name.startswith('Ar_'):continue
   g=(s>=.015)&(s<=.16)&np.isfinite(a[:,col])&np.isfinite(b[:,col]);d=a[g,col]-b[g,col]
   r['fields'][field]=dict(L2_percent=float(100*np.linalg.norm(d)/np.linalg.norm(b[g,col])),RMSE=float(np.sqrt(np.mean(d*d))))
  comparisons.append(r)
# A new physical diagnostic: chronological block markers, not independent sample error bars.
times=np.arange(2.2,4.01,.2)
fig,ax=plt.subplots(2,3,figsize=(13.8,7.4),layout='constrained')
panels=[('Ar: mesh control',[('Ar_grid_L4','L4'),('Ar_grid_L5','L5')]),('Ar: time-step control',[('Ar_grid_L5','5 ns'),('Ar_dt_dt2p5ns','2.5 ns')]),('N2: mesh control',[('N2_grid_L4','L4'),('N2_grid_L5','L5')])]
for col,(title,cases) in enumerate(panels):
 for row,angle in enumerate(['170','180']):
  for (case,label),color in zip(cases,['#777777','#1269ad']):
   v=reports[case]['profiles'][angle]['block_markers_m'];assert len(v)==10 and all(x is not None for x in v)
   ax[row,col].plot(times,np.array(v)*1000,'o-',color=color,label=label,lw=1.3,ms=4)
  ax[row,col].set(xlabel='Block end time (ms)',ylabel='Density outer marker (mm)',title=title+' / '+angle+' deg')
  ax[row,col].grid(alpha=.2);ax[row,col].legend(frameon=False)
fig.suptitle('Ten chronological full blocks, 2-4 ms: temporal variability; lines are not confidence intervals',fontsize=12)
fig.savefig(O/'Completed_controls_block_markers.png',dpi=170);fig.savefig(O/'Completed_controls_block_markers.pdf');plt.close(fig)
# Signed differences show where changes occur, preserving the original spatial coordinates.
fig,ax=plt.subplots(3,3,figsize=(13.5,10),layout='constrained')
pairs=[('Ar mesh: L4 - L5','Ar_grid_L4','Ar_grid_L5'),('Ar time step: 5 - 2.5 ns','Ar_grid_L5','Ar_dt_dt2p5ns'),('N2 mesh: L4 - L5','N2_grid_L4','N2_grid_L5')]
for row,(title,base,ref) in enumerate(pairs):
 for angle,color in [('170','#1269ad'),('180','#cf6a18')]:
  s,_,a=banks[base,angle];_,_,b=banks[ref,angle];g=(s>0)&(s<=.16)
  for col,label in enumerate(['Density difference / upstream density','Ttr difference (K)','ux difference (m/s)']):
   ax[row,col].plot(1000*s[g],a[g,col]-b[g,col],lw=1.1,color=color,label=angle+' deg')
   ax[row,col].set(xlabel='Distance outward from wall, s (mm)',ylabel=label,title=title);ax[row,col].grid(alpha=.18);ax[row,col].axhline(0,color='#999999',lw=.6);ax[row,col].legend(frameon=False)
fig.suptitle('Signed mean-profile differences in physical coordinates; no fitted shift or amplitude',fontsize=12)
fig.savefig(O/'Completed_controls_signed_differences.png',dpi=170);fig.savefig(O/'Completed_controls_signed_differences.pdf');plt.close(fig)
out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose='User-requested integrated analysis of completed Ar mesh, Ar time-step, and N2 ppc40 mesh controls',unique_complete_cases=7,comparisons=comparisons,block_diagnostics=stats,source_checks=checks,provenance=archives,scientific_acceptance=False,remaining_controls=['N2 ppc80: no mature sample blocks yet','N2 dt2.5ns: no mature sample blocks yet'],diagnostic_definitions=dict(between_block_RMS='100*sqrt(mean_b(sum_x((q_b-qmean)^2)))/norm(qmean)',half_drift='100*norm(mean(first5)-mean(last5))/norm(all10mean)',caution='Finite-window descriptive diagnostics only; spatial locations and chronological blocks are not independent replicates. No p-values, CIs, effective sample counts or automatic steady-state acceptance inferred.'))
(O/'ASSESSMENT.json').write_text(json.dumps(out,indent=2))
print('COMPARISONS',json.dumps(comparisons))
print('L5_BLOCK_DIAGNOSTICS',json.dumps([r for r in stats if r['case'] in ['Ar_grid_L5','Ar_dt_dt2p5ns','N2_grid_L5']]))
