
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,hashlib,csv,importlib.util,datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path('campaign');O=W/'editorial_current_revision';P=W/'pod_all_kn_v12/result_v14_20261025T025404Z';E=O/'evidence';F=O/'manuscript/figures'
S=json.loads((P/'SUMMARY.json').read_text());rows=list(csv.DictReader((W/'pod_all_kn_v12/SELECTED_EXACT_KN.csv').open()))
spec=importlib.util.spec_from_file_location('reader',W/'source/cylinder_step3_shock_layer_metrics.py');rd=importlib.util.module_from_spec(spec);spec.loader.exec_module(rd)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
control=[];heldout=[];inputs={};profiledata={}
def bilinear(q,s,bounds):
 xmin,xmax,ymin,ymax=bounds;xg=np.linspace(xmin,xmax,480);yg=np.linspace(ymin,ymax,320)
 theta=np.deg2rad(170);x=.1524+(.1524+s)*np.cos(theta);y=(.1524+s)*np.sin(theta)
 u=(x-xmin)/(xmax-xmin)*479;v=(y-ymin)/(ymax-ymin)*319
 out=np.full(len(s),np.nan);g=(u>=0)&(u<479)&(v>=0)&(v<319);ii=np.floor(u[g]).astype(int);jj=np.floor(v[g]).astype(int);du=u[g]-ii;dv=v[g]-jj
 values=np.array([q[jj,ii],q[jj,ii+1],q[jj+1,ii],q[jj+1,ii+1]])
 out[g]=(values*np.array([(1-du)*(1-dv),du*(1-dv),(1-du)*dv,du*dv])).sum(axis=0)
 return out
plt.rcParams.update({'font.size':15,'axes.labelsize':17,'legend.fontsize':12,'pdf.fonttype':42,'lines.linewidth':1.8})
colors={'rho':'#0072B2','ttr':'#D55E00','trot':'#7B3294','u':'#555555'}
labels={'rho':r'$\rho/\rho_\infty$','ttr':r'$T_{\rm tr}/T_\infty$','trot':r'$T_{\rm rot}/T_\infty$','u':r'$u_x/U_\infty$'}
for gas,fam in [('Ar','all_Kn_historical'),('N2','historical_only')]:
 cases=[r for r in rows if r['gas']==gas and (gas=='Ar' or float(r['kn'])!=.075)]
 bounds=[]
 for r in cases:
  for k in ['kn','mach','nrho_in','T_in','U_in']:r[k]=float(r[k])
  d=rd.read_case(r);bounds.append([float(d['x'].min()),float(d['x'].max()),float(d['y'].min()),float(d['y'].max())]);inputs[r['path']]=sha(r['path'])
 b=[max(x[0] for x in bounds),min(x[1] for x in bounds),max(x[2] for x in bounds),min(x[3] for x in bounds)]
 variables=['rho','ttr','u']+(['trot'] if gas=='N2' else [])
 fig,axs=plt.subplots(1,len(variables),figsize=(12,4.1));s=np.linspace(.015,.16,1451)
 for var in ['rho','ttr','u','p','mach_proxy']+(['trot'] if gas=='N2' else []):
  p=P/f'{gas}_{var}_physical_480_{fam}.npz';a=P/f'{gas}_{var}_attached_sigma1.0.npz';z=np.load(p);za=np.load(a);inputs[str(p)]=sha(p);inputs[str(a)]=sha(a)
  X=z['mean']+z['coefficients']@z['modes'];w=z['weights'];sw=np.sqrt(w);kn=z['kn'];subset=np.array([np.any(np.isclose(k,za['kn'],rtol=0,atol=1e-10)) for k in kn])
  ref=next(r for r in S if r['gas']==gas and r['variable']==var and r['basis']=='physical' and r.get('grid')==[480,320] and r.get('family')==fam)
  ss=np.linalg.svd((X-X.mean(axis=0))*sw,compute_uv=False);e=ss*ss/np.sum(ss*ss);assert abs(e[0]-ref['E1'])<1e-10
  xs=X[subset];sv=np.linalg.svd((xs-xs.mean(axis=0))*sw,compute_uv=False);es=sv*sv/np.sum(sv*sv)
  ar=next(r for r in S if r['gas']==gas and r['variable']==var and r['basis']=='attached' and r.get('sigma_factor')==1.0) if any(r.get('sigma_factor')==1.0 and r.get('basis')=='attached' and r['gas']==gas and r['variable']==var for r in S) else None
  # Attached spectra are independently read from their exact cached singular values.
  ae=za['S']**2/np.sum(za['S']**2)
  control.append(dict(gas=gas,variable=var,full_kn=kn.tolist(),retained_kn=kn[subset].tolist(),physical_points=int(len(w)),physical_full_E1=float(e[0]),physical_retained_E1=float(es[0]),physical_retained_C2=float(es[:2].sum()),attached_E1=float(ae[0]),attached_points=int(za['support'].sum()),same_physical_features_and_weights=True,attached_support_and_norm_differ=True))
  if var not in variables:continue
  test=int(np.flatnonzero(np.isclose(kn,.1))[0]);train=np.arange(len(kn))!=test;mu=X[train].mean(axis=0);zt=(X[train]-mu)*sw;_,_,v=np.linalg.svd(zt,full_matrices=False)
  k=np.log(kn[train]);order=np.argsort(k);ct=zt@v[:2].T;cp=np.array([np.interp(np.log(.1),k[order],ct[order,j]) for j in range(2)]);pred=mu+(cp@v[:2])/sw
  err=float(100*np.linalg.norm((pred-X[test])*sw)/np.linalg.norm(X[test]*sw))
  referr=next(r['logKn_coefficient_interpolation_L2_percent'] for r in ref['leave_one_out'] if r['r']==2 and abs(r['heldout_Kn']-.1)<1e-12);assert abs(err-referr)<1e-8,(gas,var,err,referr)
  fields=[]
  for q in [X[test],pred]:
   full=np.full(153600,np.nan);full[z['support'].astype(bool)]=q;fields.append(bilinear(full.reshape(320,480),s,b))
  good=np.isfinite(fields[0])&np.isfinite(fields[1]);assert good.sum()>100
  # Preserve any original missing coverage in both plotted curves.
  for q in fields:q[~good]=np.nan
  ax=axs[variables.index(var)];ax.plot(s*1000,fields[0],color=colors[var],label='Held-out DS2V');ax.plot(s*1000,fields[1],color=colors[var],ls='--',label='Rank-two interpolation');ax.set(xlim=(15,160),ylabel=labels[var])
  values=np.concatenate([q[good] for q in fields]);lo,hi=float(values.min()),float(values.max());pad=.06*(hi-lo);ax.set_ylim(lo-pad,hi+pad)
  profiledata[gas+'_'+var+'_true']=fields[0];profiledata[gas+'_'+var+'_prediction']=fields[1]
  heldout.append(dict(gas=gas,variable=var,heldout_kn=.1,rank=2,training_kn=kn[train].tolist(),weighted_full_field_error_percent=err,previous_cached_error_percent=referr,cache_equivalence_passed=True,ray170_finite_points=int(good.sum()),physical_domain_bounds=b,normalization='Nominal upstream values; weighted full-heldout-field denominator, not a percentage of fluctuation.'))
 fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.99),ncol=2)
 fig.supxlabel('Wall distance s (mm), increasing upstream',fontsize=17,y=.01);fig.tight_layout(rect=(0,.12,1,.85))
 for ext in ['pdf','png']:fig.savefig(F/f'POD_heldout_{gas}_M10_Kn01.{ext}',dpi=160,bbox_inches='tight')
 plt.close(fig)
profiledata['s_m']=s;np.savez_compressed(E/'HELDOUT_PREDICTION_PROFILES.npz',**profiledata)
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),control=control,heldout=heldout,inputs_sha256=inputs,method='Recover exact full-rank cached normalized physical fields; hold original full-ensemble physical features/weights fixed while restricting case list. The attached comparison changes support/norm and is not a registration-gain measurement. Heldout Kn.1 profiles reuse exact training-only rank2 logKn coefficient interpolation and match cached errors.',no_new_solver_job=True,no_new_acceptance_threshold=True)
(E/'REPRESENTATION_CONTROL.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(control=control,heldout=heldout)),flush=True)
