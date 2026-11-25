from pathlib import Path
import json,csv,ast,importlib.util,hashlib,datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.tri as mtri
W=Path('campaign')
O=W/'editorial_current_revision/evidence/restored_mach_pod';O.mkdir(exist_ok=True,parents=True)
R=.1524
spec=importlib.util.spec_from_file_location('reader',W/'source/cylinder_step3_shock_layer_metrics.py');reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
src=(W/'pod_all_kn_v14.py').read_text();tree=ast.parse(src);names=['smooth','outer','masked_triangulation','interp','resample','pod']
for node in tree.body:
 if isinstance(node,ast.FunctionDef) and node.name in names:
  code=ast.get_source_segment(src,node).replace("'heldout_Kn'","'heldout_Mach'").replace("'logKn_coefficient_interpolation_L2_percent'","'logMach_coefficient_interpolation_L2_percent'")
  exec(code,globals())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
case_lambda=.003048
rows=[r for r in csv.DictReader((W/'selected_cases.csv').open()) if r['sweep']=='Mach' and float(r['mach']).is_integer()]
rows=sorted(rows,key=lambda r:(r['gas'],float(r['mach'])))
ex=json.loads((W/'reviewer_figures_v4_20261031/EXTRACTION.json').read_text())
references={g:next(r['rhoinf'] for r in ex if r['gas']==g and r['sweep']=='Mach' and r['marker']) for g in ['Ar','N2']}
summary=[];provenance=[];markers=[];rays={}
angles=np.linspace(100,180,65);xi=np.linspace(-1,3,301)
for gas in ['Ar','N2']:
 rr=[r for r in rows if r['gas']==gas];loaded=[]
 for r in rr:
  data=reader.read_case(r);loaded.append((r,data));provenance.append(dict(path=r['path'],mach=float(r['mach']),gas=gas,sha256=sha(r['path'])))
 xmin=max(d['x'].min() for _,d in loaded);xmax=min(d['x'].max() for _,d in loaded);ymin=max(d['y'].min() for _,d in loaded);ymax=min(d['y'].max() for _,d in loaded)
 X,Y=np.meshgrid(np.linspace(xmin,xmax,240),np.linspace(ymin,ymax,160));wx=np.ones(240);wy=np.ones(160);wx[[0,-1]]=.5;wy[[0,-1]]=.5;weight=np.outer(wy,wx).ravel()
 physical={};attached={};metadata=[]
 variables=['rho','u','p','ttr']+(['trot'] if gas=='N2' else [])
 for row,d in loaded:
  M=float(row['mach']);mass=6.63e-26 if gas=='Ar' else 4.65e-26;gamma=5/3 if gas=='Ar' else 1.4
  d['mach_proxy']=np.sqrt(d['u']**2+d['v']**2+d['w']**2)/np.sqrt(np.maximum(gamma*1.380649e-23*d['ttr']/mass,1e-30))
  vs=variables+['mach_proxy'];tr=masked_triangulation(d)
  # Fixed same-gas scalar factors, independent of the varied Mach parameter.
  scales=dict(rho=references[gas],u=1.,p=1.,ttr=200.,trot=200.,mach_proxy=1.)
  ips={v:mtri.LinearTriInterpolator(tr,d[v]/scales[v]) for v in vs}
  for v,ip in ips.items():physical.setdefault(v,[]).append(interp(ip,X,Y).ravel())
  line=np.abs(d['y']-d['y'].min())<1e-8;xx=np.unique(d['x'][line&(d['x']<0)]);spacing=float(np.median(np.diff(xx))) if len(xx)>10 else .0022
  sigma=max(.5*spacing,.01*case_lambda);snaps={v:np.full((65,301),np.nan) for v in vs};jac=np.full((65,301),np.nan);supported=0
  for ia,a in enumerate(angles):
   direction=np.array([np.cos(np.deg2rad(a)),np.sin(np.deg2rad(a))]);origin=np.array([R,0])+R*direction;limits=[]
   for k,(lo,hi) in enumerate([(d['x'].min(),d['x'].max()),(d['y'].min(),d['y'].max())]):
    if abs(direction[k])>1e-12:
     t=((hi if direction[k]>0 else lo)-origin[k])/direction[k]
     if t>0:limits.append(t)
   if not limits:continue
   s=np.linspace(0,min(limits),1601);q={v:interp(ip,origin[0]+s*direction[0],origin[1]+s*direction[1]) for v,ip in ips.items()};m=outer(s,q['rho'],sigma)
   markers.append(dict(gas=gas,mach=M,angle=float(a),supported=m is not None,**(m or {})))
   if ia==56:rays[(gas,M)]=(s,q,m)
   if m is None:continue
   supported+=1;t=m['s']+xi*m['delta'];jac[ia]=(R+t)*m['delta']
   for v in vs:snaps[v][ia]=resample(s,q[v],t)
  metadata.append(dict(mach=M,supported_rays=supported))
  for v in vs:attached.setdefault(v,[]).append((snaps[v].ravel(),jac.ravel(),supported))
  print('MACH_SAMPLED',gas,M,supported,flush=True)
 values=[float(r['mach']) for r in rr]
 for v in vs:
  r=pod(np.array(physical[v]),weight,values,f'{gas}_{v}_physical_Mach')
  summary.append(dict(gas=gas,variable=v,basis='physical',grid=[240,160],parameter='Mach',**r))
  group=attached[v];keep=[i for i,g in enumerate(group) if g[2]>=8]
  if len(keep)>=3:
   support=np.all(np.isfinite(np.array([group[i][0] for i in keep])),axis=0);w=np.nanmean(np.array([group[i][1] for i in keep]),axis=0);w[~support]=0
   r=pod(np.array([group[i][0] for i in keep]),w,[values[i] for i in keep],f'{gas}_{v}_attached_Mach')
   summary.append(dict(gas=gas,variable=v,basis='density_attached',parameter='Mach',excluded_Mach=[values[i] for i in range(len(values)) if i not in keep],**r))
 for (g,M),(s,q,m) in list(rays.items()):
  if g==gas:np.savez_compressed(O/f'{g}_Mach{M:g}_ray170.npz',s=s,**q)
(O/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
(O/'MARKERS.json').write_text(json.dumps(markers,indent=2))
(O/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2))
method=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source='original submitted Mach sweep fields',selected_Ar_Mach=list(range(5,16)),selected_N2_Mach=list(range(5,15)),available_extra_Ar_fractional_Mach_not_used=True,physical_grid=[240,160],normalization='One fixed scalar factor per gas and field, not case-wise heated-bulk normalization; fixed density reference from archived extraction. Scalar variance fractions invariant to constant scalar scale.',attached='Strict finite common support, outer compression peak, no imputation; minimum8supportedrays selection, fixed mean radial Jacobian on retained cases',no_new_solver_run=True,training_only_mean_each_loo=True,scalar_variance_not_kinetic_energy=True,source_script_sha256=sha(__file__))
(O/'METHOD.json').write_text(json.dumps(method,indent=2))
print('MACH_ANALYSIS_COMPLETE',json.dumps([dict(gas=r['gas'],variable=r['variable'],basis=r['basis'],E1=r.get('E1'),n=r.get('snapshots'),points=r.get('common_points')) for r in summary]),flush=True)
