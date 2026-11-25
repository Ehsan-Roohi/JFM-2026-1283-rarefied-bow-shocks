"""Existing-field numerical reconciliation. No solver, jobs, or live files are modified."""
from pathlib import Path
import sys, ast, csv, json, hashlib, importlib.util, datetime, math, subprocess
sys.dont_write_bytecode=True
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
W=Path('campaign')
O=W/'editorial_scientific_reconciliation_20261117/numerical_candidate'
O.mkdir(parents=True,exist_ok=True)
R=.1524
case_lambda=.003048
P=W/'pod_all_kn_v12/result_v14_20261025T025404Z'
B=Path('DS2V_source')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readcsv(p):return list(csv.DictReader(Path(p).open()))
def savecsv(name,rows):
 with (O/name).open('w',newline='') as h:
  writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
sp=importlib.util.spec_from_file_location('audited_reader',W/'source/cylinder_step3_shock_layer_metrics.py')
rd=importlib.util.module_from_spec(sp);sp.loader.exec_module(rd)
source=W/'pod_all_kn_v14.py'
tree=ast.parse(source.read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['smooth','outer','masked_triangulation','interp']],type_ignores=[]),str(source),'exec'))
kn=readcsv(W/'pod_all_kn_v12/SELECTED_EXACT_KN.csv')
mach=[r for r in readcsv(W/'selected_cases.csv') if r['sweep']=='Mach']
oldex=json.loads((W/'reviewer_figures_v4_20261031/EXTRACTION.json').read_text())
oldmarkers=readcsv(P/'MARKERS.csv')
def geometry(data):
 line=(np.abs(data['y']-data['y'].min())<1e-8)&(data['x']<0)
 xx=np.unique(data['x'][line])
 spacing=float(np.median(np.diff(xx))) if len(xx)>10 else float(np.sqrt(np.ptp(data['x'])*np.ptp(data['y'])/len(data['x'])))
 return masked_triangulation(data),spacing

def ray(data,tri,angle,n=1601):
 vec=np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))]);origin=np.array([R,0])+R*vec;lim=[]
 for k,nm in enumerate(['x','y']):
  if abs(vec[k])>1e-12:
   t=((data[nm].max() if vec[k]>0 else data[nm].min())-origin[k])/vec[k]
   if t>0:lim.append(t)
 if not lim:return None,None,None
 s=np.linspace(0,min(lim),n);xy=origin[:,None]+vec[:,None]*s
 rho=interp(mtri.LinearTriInterpolator(tri,data['rho']),xy[0],xy[1])
 return s,rho,xy

def getreference(data,tri):
 s,rho,_=ray(data,tri,170)
 tail=(s>s[-1]-.035)&(s<s[-1]-.005)
 return float(np.nanmedian(rho[tail]))

cases=[];matrices=[];ret={}
for bank,rows in [('Kn',[r for r in kn if float(r['kn'])==.01]),('Mach',mach)]:
 for row in rows:
  gas=row['gas'];d=rd.read_case(row);tri,spacing=geometry(d);case_lambda=float(row['kn'])*2*R
  mass=6.63e-26 if gas=='Ar' else 4.65e-26
  exact=next(r for r in kn if r['gas']==gas and float(r['kn'])==.01)
  nominal=float(exact['nrho_in'])*mass;measured=getreference(d,tri)
  reference=float(row['nrho_in'])*mass if bank=='Kn' else measured
  s,rho,_=ray(d,tri,170);sig=max(.5*spacing,.01*case_lambda);mark=outer(s,rho/reference,sig)
  old=(next((q for q in oldex if q['gas']==gas and q['sweep']=='Mach' and q['parameter']==float(row['mach'])),None) if bank=='Mach' else None)
  rec={'bank':bank,'gas':gas,'mach':float(row['mach']),'kn_label':float(row['kn']),'field_path':row['path'],'field_sha256':sha(row['path']),'field_bytes':Path(row['path']).stat().st_size,'exported_points':len(d['x']),'exported_domain':{k:[float(d[k].min()),float(d[k].max())] for k in ['x','y']},'exported_spacing_proxy_m':spacing,'rho_reference':reference,'rho_reference_kind':'imposed matching DS2VD input' if bank=='Kn' else 'measured 170degree ray last5to35mm window; matching Mach input absent','measured_to_other_exact_Kn_nominal':measured/nominal,'marker':mark,'prior_v4_marker':old['marker'] if old else None}
  cases.append(rec)
  if float(row['mach'])==10:
   ret[gas,bank]=(row,d,tri,spacing,rec)
   for n,sm in [(1601,'median_upstream_x'),(2001,'area_per_point')]:
    ss,rr,_=ray(d,tri,170,n);tail=(ss>ss[-1]-.035)&(ss<ss[-1]-.005);meas=float(np.nanmedian(rr[tail]));sigma=max(.5*(spacing if n==1601 else np.sqrt(np.ptp(d['x'])*np.ptp(d['y'])/len(d['x']))),.01*case_lambda)
    for mode,norm in [('matching_Kn_nominal' if bank=='Kn' else 'other_bank_Kn_nominal_for_sensitivity_only',nominal),('measured_Mach_tail',meas)]:matrices.append({'gas':gas,'bank':bank,'n_samples':n,'spacing_method':sm,'normalization':mode,'rho_reference':norm,'marker':outer(ss,rr/norm,sigma)})
  print('AUDITED',bank,gas,row['mach'],bool(mark),flush=True)

# Frozen exact-Kn primary markers reproduced to floating-point tolerance.
primary_retests=[]
for gas in ['Ar','N2']:
 row,d,tri,spacing,rec=ret[gas,'Kn']
 previous=next(r for r in oldmarkers if r['gas']==gas and float(r['kn'])==.01 and float(r['angle'])==170 and float(r['smoothing_factor'])==1)
 test={k:abs(rec['marker'][k]-float(previous[k])) for k in ['s','delta','downstream_s','sigma']}
 assert max(test.values())<1e-12
 primary_retests.append({'gas':gas,'absolute_error_m':test})

# Corrected Fig2 ridge: same actual Mach field as original SWD; no rigid shift.
origpath=B/'Shock_Compare_Akhlaghi_v22_fixed_biglegend/shock_compare_points.csv'
orig=[r for r in readcsv(origpath) if r['gas']=='Ar' and float(r['Kn'])==.01 and float(r['Mach'])==10]
for q in orig:
 q['theta']=math.degrees(math.atan2(float(q['y']),float(q['x'])-R));q['s_actual']=math.hypot(float(q['x'])-R,float(q['y']))-R
legacy=sorted([q for q in orig if q['method']=='ray_center'],key=lambda q:q['theta'])
swd=sorted([q for q in orig if q['method']=='akhlaghi_sigma1'],key=lambda q:q['theta'])
row,d,tri,spacing,rec=ret['Ar','Mach'];case_lambda=.003048
angles=sorted(set([q['theta'] for q in legacy]+[170.0]));ridge=[];comparison=[]
for angle in angles:
 s,rho,_=ray(d,tri,angle);marker=outer(s,rho/rec['rho_reference'],max(.5*spacing,.01*case_lambda))
 if marker:
  v=np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))]);xy=np.array([R,0])+(R+marker['s'])*v
  xs,ys=map(float,xy);ms=marker['s'];delta=marker['delta']
 else:xs=ys=ms=delta=float('nan')
 item={'angle_deg':angle,'supported':marker is not None,'x_m':xs,'y_m':ys,'s_m':ms,'delta_m':delta,'rho_reference':rec['rho_reference']};ridge.append(item)
 old_s=float(np.interp(angle,[q['theta'] for q in legacy],[q['s_actual'] for q in legacy])) if legacy[0]['theta']<=angle<=legacy[-1]['theta'] else None
 swd_s=float(np.interp(angle,[q['theta'] for q in swd],[q['s_actual'] for q in swd])) if swd[0]['theta']<=angle<=swd[-1]['theta'] else None
 comparison.append(dict(item,original_edge_s_m=old_s,swd_sigma1_interpolated_s_m=swd_s,new_minus_swd_mm=(ms-swd_s)*1000 if marker and swd_s is not None else None,old_minus_new_mm=(old_s-ms)*1000 if marker and old_s is not None else None))
savecsv('SWD_CORRECTED_RIDGE_POINTS.csv',ridge);savecsv('SWD_BEFORE_AFTER_POINTS.csv',comparison)
# Preserve original source coordinates exactly, compact local source table only.
(O/'ORIGINAL_SHOCK_COMPARISON_POINTS.csv').write_bytes(origpath.read_bytes())

plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':12,'axes.labelsize':12,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.8})
outputs=[]
def save(fig,name):
 for ext in ['pdf','png']:
  p=O/(name+'.'+ext);fig.savefig(p,dpi=240,bbox_inches='tight');outputs.append({'file':p.name,'bytes':p.stat().st_size,'sha256':sha(p)})
 plt.close(fig)
fig,ax=plt.subplots(figsize=(5.4,3.6),layout='constrained')
im=ax.tricontourf(tri,d['rho']/rec['rho_reference'],levels=np.linspace(0,30,151),cmap='turbo',extend='neither')
try:im.set_rasterized(True)
except AttributeError:
 for q in im.collections:q.set_rasterized(True)
ax.add_patch(plt.Circle((R,0),R,fc='white',ec='black',lw=1))
ax.plot([q['x_m'] for q in ridge],[q['y_m'] for q in ridge],color='black',lw=1.2,label='Outer density peak')
display=[]
for method,col,marker,lab in [('akhlaghi_sigma1','#9d248f','o',r'SWD, $\sigma=1$'),('akhlaghi_sigma0.1','#00a8bf','s',r'SWD, $\sigma=0.1$')]:
 rr=sorted([q for q in orig if q['method']==method],key=lambda q:float(q['y']));idx=sorted(set(list(range(0,len(rr),8))+[len(rr)-1]));show=[rr[i] for i in idx]
 ax.plot([float(q['x']) for q in show],[float(q['y']) for q in show],color=col,ls='None',marker=marker,ms=3.2,mec=col,mew=.2,label=lab)
 display.append({'method':method,'all_rows':len(rr),'display_indices':idx,'display_rows':len(show),'omitted_for_visual_spacing_only':True})
ax.set(xlim=(float(d['x'].min()),float(d['x'].max())),ylim=(0,float(d['y'].max())),xlabel='$x$ (m)',ylabel='$y$ (m)');ax.set_aspect('equal');ax.tick_params(direction='in',top=True,right=True)
ax.legend(loc='upper right',frameon=True,facecolor='white',framealpha=.97,edgecolor='.8',borderpad=.35,handlelength=1.6,labelspacing=.25)
cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.025);cb.set_label(r'$\rho/\rho_{\mathrm{ref}}$ (measured)');cb.set_ticks([0,10,20,30]);save(fig,'compare_Ar_Kn0p01_M10p0_mach_sweep_argon')

fig,axs=plt.subplots(1,2,figsize=(7.0,3.2),layout='constrained')
for gas,marker,lab in [('Ar','o','Ar'),('N2','s',r'N$_2$')]:
 rows=sorted([q for q in cases if q['bank']=='Mach' and q['gas']==gas and q['marker']],key=lambda q:q['mach'])
 for ax,key in zip(axs,['s','delta']):ax.plot([q['mach'] for q in rows],[q['marker'][key]*1000 for q in rows],color='black',ls='None',marker=marker,ms=4.2,mfc='white' if gas=='N2' else 'black',mew=.9,label=lab)
for ax,lab in zip(axs,['Outer density location (mm)','Local density width (mm)']):ax.set(xlabel=r'$M_\infty$',ylabel=lab,xticks=[5,10,15]);ax.tick_params(direction='in',top=True,right=True);ax.legend(frameon=False)
save(fig,'Step3_Mach_Ar_N2_density_metrics')

snapshot=next(q for q in comparison if q['angle_deg']==170)
summary={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Read-only existing DS2V field analysis and staging-only figures; no solver, simulation, jobs, checkpoint or manuscript changes','source_functions':{'path':str(source),'sha256':sha(source),'functions':['smooth','outer','masked_triangulation','interp']},'reader':{'path':str(W/'source/cylinder_step3_shock_layer_metrics.py'),'sha256':sha(W/'source/cylinder_step3_shock_layer_metrics.py')},'shared_primary_definition':'v14 outermost resolved compression maximum; 1601 samples per ray, Gaussian sigma=max(0.5 upstream-x exported spacing,0.01 KnD), existing plateau/prominence guards unchanged','normalization':'Kn uses matching DS2VD nominal density; Mach uses its measured170degree ray final5â€“35mm window because original Mach input unavailable. This is not physical model equivalence.','kn_primary_retests':primary_retests,'cases':cases,'cross_pipeline_matrix':matrices,'swd_original':{'path':str(origpath),'sha256':sha(origpath),'method_source':str(B/'compare_akhlaghi_vs_ray_shock_v22_fixed.py'),'method_source_sha256':sha(B/'compare_akhlaghi_vs_ray_shock_v22_fixed.py'),'original_edge_algorithm_lines':'452â€“459: farthest sample >=0.2*maximum |density gradient|, then localÂ±4sample search; not corrected peak','legacy_ridge_rows':len(legacy),'swd_center_rows':len(swd),'new_supported_rays':sum(q['supported'] for q in ridge),'requested_angles':len(ridge),'at_170deg':snapshot,'display_subset':display},'outputs':outputs,'uncertainties':['Mach original DS2VD input was not found in matching extracted folders; full ArMach RAR21members and resultsN2 RAR40members list no input file. Fields cannot verify imposed density or collision model.','Exported point grid is not collision-cell resolution.','SWD distances interpolated only between actual bracketing center points; not a new solver measurement.','New detector settings unify numerical primary definition, not bank-specific raw fields or physical configurations.']}
(O/'NUMERICAL_PROVENANCE_AUDIT.json').write_text(json.dumps(summary,indent=2,allow_nan=False,default=lambda x:float(x))+'\n')
flat=[]
for q in cases:
 m=q['marker'];old=q['prior_v4_marker'];flat.append({'bank':q['bank'],'gas':q['gas'],'mach':q['mach'],'field_sha256':q['field_sha256'],'supported':m is not None,'s_mm':m['s']*1000 if m else None,'delta_mm':m['delta']*1000 if m else None,'prior_s_mm':old['s']*1000 if old else None,'prior_delta_mm':old['delta']*1000 if old else None,'rho_reference':q['rho_reference'],'reference_kind':q['rho_reference_kind']})
savecsv('DENSITY_METRICS_PRIMARY.csv',flat)
print('FINISHED',json.dumps({'path':str(O),'cases':len(cases),'supported_rays':sum(q['supported'] for q in ridge),'at170':snapshot,'outputs':outputs}),flush=True)