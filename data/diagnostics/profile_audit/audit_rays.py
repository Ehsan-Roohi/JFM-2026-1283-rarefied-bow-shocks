"""Finite interpolated DSMC ray intersections with imposed ideal jump levels and detector sensitivity."""
from pathlib import Path
import json,csv,ast,hashlib
import numpy as np
root=Path(__file__).resolve().parent
project=Path(r'campaign/editorial_current_revision')
R=.1524
reference=project/'review/scientific_reconciliation/NUMERICAL_PROVENANCE_AUDIT.json'
original=json.loads(reference.read_text())
source=next((project/'evidence/figure_inputs').glob('*pod_all_kn_v14.py'))
tree=ast.parse(source.read_text())
names=['smooth','outer']
defs=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names];assert len(defs)==len(names)
exec(compile(ast.Module(body=defs,type_ignores=[]),str(source),'exec'))
archive=np.load(root/'existing_density_rays.npz');meta=json.loads(str(archive['meta']))
def writecsv(filename,rows):
 with (root/filename).open('w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
peakrows=[];crossrows=[];paired=[];same_sigma=[];sensitivity=[];snapshot=[]
cached=list(csv.DictReader((project/'evidence/MARKERS.csv').open()))
four=list(csv.DictReader((project/'review/scientific_reconciliation/PRIMARY_FOUR_DENSITY_PROFILES.csv').open()))
def segment(s,r):
 finite=np.flatnonzero(np.isfinite(r))
 regions=np.split(finite,np.flatnonzero(np.diff(finite)>1)+1)
 ii=regions[-1]
 assert len(ii)>200
 return s[ii],r[ii],len(regions)
def crossed(s,v,threshold):
 # s increases outward; first crossing from upstream is the outermost decreasing outward.
 ix=np.flatnonzero((v[1:]<threshold)&(v[:-1]>=threshold))
 roots=s[ix]+(threshold-v[ix])*(s[ix+1]-s[ix])/(v[ix+1]-v[ix])
 exists=len(roots)>0 and v[-1]<threshold
 first=float(roots[-1]) if exists else None
 why='bracketed' if exists else ('upstream-at-or-above-threshold' if v[-1]>=threshold else 'maximum-below-threshold' if max(v)<threshold else 'no-inward-crossing')
 return first,len(ix),why,roots
def maybe(x):
 return float(x)*1000 if x is not None else None
for c in meta['cases']:
 gas=c['gas'];bank=c['bank'];kn=c['kn'];case_lambda=.3048*kn
 gamma=5/3 if gas=='Ar' else 1.4;rh=(gamma+1)*100/((gamma-1)*100+2)
 key=c['prefix'];s0=archive[key+'_s'];rho0=archive[key+'_rho']/c['rho_ref'];h=c['spacing_m']
 sig=max(.5*h,.01*case_lambda)
 s,raw,nsegments=segment(s0[::4],rho0[::4])
 ds=s[1]-s[0]
 if bank=='Kn' and kn in [.01,.025]:
  fro=[v for v in four if v['gas']==gas and float(v['Kn'])==kn]
  np.testing.assert_allclose(s,[float(z['s_m']) for z in fro],rtol=0,atol=1e-14)
  np.testing.assert_allclose(raw,[float(z['raw_rho_over_rhoinf']) for z in fro],rtol=1e-13,atol=1e-14)
  for x,q in zip(s,raw):snapshot.append({'bank':bank,'gas':gas,'Kn_or_nominal_Kn':kn,'angle_deg':170,'s_m':x,'rho_kg_m3':q*c['rho_ref'],'rho_over_ref':q,'reference_kg_m3':c['rho_ref'],'reference_kind':c['rho_ref_kind'],'export_x_spacing_m':h})
 if bank=='Mach':
  for x,q in zip(s,raw):snapshot.append({'bank':bank,'gas':gas,'Kn_or_nominal_Kn':kn,'angle_deg':170,'s_m':x,'rho_kg_m3':q*c['rho_ref'],'rho_over_ref':q,'reference_kg_m3':c['rho_ref'],'reference_kind':c['rho_ref_kind'],'export_x_spacing_m':h})
 if kn==.01:
  for width in [.0005,.001095,.00219,.00438]:
   for normal,scale in [('canonical',1),('measured_tail',c['rho_ref']/c['rho_tail'])]:
    mark=outer(s,raw*scale,width)
    same_sigma.append({'gas':gas,'bank':bank,'normalization':normal,'sigma_mm':width*1000,'peak_mm':maybe(mark['s']) if mark else None,'delta_localjump_mm':maybe(mark['delta']) if mark else None})
 if bank!='Kn':continue
 for stride in [8,4,2,1]:
  ns=len(s0[::stride])
  s,r,nseg=segment(s0[::stride],rho0[::stride]);ds=s[1]-s[0]
  for fac in [0.,.5,1.,2.]:
   q=smooth(r,ds,sig*fac) if fac else r
   tail=q[-max(20,len(q)//10):];med=float(np.median(tail));spread=float(np.std(tail)/med);plateau=abs(1-med)<=.1 and spread<=.1
   rec={'gas':gas,'Kn':kn,'factor':fac,'ray_points':ns,'sigma_mm':fac*sig*1000,'out_h_mm':1000*h,'outermost_finite_s_min_mm':s[0]*1000,'outermost_finite_s_max_mm':s[-1]*1000,'finite_components':nseg,'upstream_sample':q[-1],'tail_median':med,'tail_rel_sd':spread,'upstream_plateau_pass':plateau,'max_rho_ratio':max(q)}
   for pct in [10,50,90]:
    val=1+pct/100*(rh-1)
    x,num,status,roots=crossed(s,q,val)
    rec[f's{pct}_mm']=maybe(x);rec[f'n{pct}']=num;rec[f'status{pct}']=status
   rec['W_mm']=rec['s10_mm']-rec['s90_mm'] if rec['s10_mm'] is not None and rec['s90_mm'] is not None else None
   rec['crossing_order_pass']=rec['s10_mm']>rec['s50_mm']>rec['s90_mm'] if all(rec[f's{i}_mm'] is not None for i in [10,50,90]) else None
   if ns==1601:crossrows.append(rec)
   if fac==1:sensitivity.append(rec)
   if fac==0:continue
   mark=outer(s,r,fac*sig)
   local_edge=None;down=None;target=None
   if mark:
    dw=np.abs(s-mark['downstream_s'])<max(fac*sig,.01*R)
    down=float(np.median(q[dw]));target=1+.1*(down-1)
    _,_,_,roots=crossed(s,q,target);valid=roots[roots>mark['s']]
    if len(valid)==1:local_edge=float(valid[0])
   mrow={'gas':gas,'Kn':kn,'factor':fac,'ray_points':ns,'sigma_mm':fac*sig*1000,'out_h_mm':1000*h,'outer_peak_supported':bool(mark),'s_peak_mm':maybe(mark['s']) if mark else None,'local_width_mm':maybe(mark['delta']) if mark else None,'s10_local_mm':maybe(local_edge),'local_down_ratio':down,'local10threshold':target,'upstream_local_offset_over_lambda':(local_edge-mark['s'])/case_lambda if local_edge is not None else None}
   peakrows.append(mrow)
   if ns==1601:
    cert=next(x for x in cached if x['gas']==gas and float(x['kn'])==kn and float(x['angle'])==170 and float(x['smoothing_factor'])==fac)
    assert (cert['supported']=='True')==bool(mark),(gas,kn,fac)
    if mark:assert abs(float(cert['s'])-mark['s'])<1e-11 and abs(float(cert['delta'])-mark['delta'])<1e-10,(gas,kn,fac)
writecsv('fixed_RH_crossings.csv',crossrows)
writecsv('ray_discretization_sensitivity.csv',sensitivity)
writecsv('outer_peak_sensitivity.csv',peakrows)
writecsv('cross_family_matched_sigma.csv',same_sigma)
writecsv('170deg_bank_comparison_rays.csv',snapshot)
primary=[x for x in crossrows if x['factor']==1]
def simple(x):return None if x is None else round(x,2)
for x in primary:
 print(x['gas'],x['Kn'],'s10/50/90, W mm',*[simple(x[k]) for k in ['s10_mm','s50_mm','s90_mm','W_mm']],'roots',*[x[k] for k in ['n10','n50','n90']],'tail',round(x['tail_median'],3),x['upstream_plateau_pass'])
for x in same_sigma:
 if x['normalization']=='canonical':print('paired',x['gas'],x['bank'],x['sigma_mm'],simple(x['peak_mm']),simple(x['delta_localjump_mm']))
for x in peakrows:
 if x['ray_points']==1601 and x['Kn']<.03:print('local',x['gas'],x['Kn'],x['factor'],simple(x['s_peak_mm']),simple(x['s10_local_mm']),simple(x['upstream_local_offset_over_lambda']))
