from pathlib import Path
import json,ast,hashlib,importlib.util,numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt,matplotlib.tri as mtri
W=Path('campaign');O=W/'reviewer_figures_v4_20261031';R=.1524;case_lambda=.003048
spec=importlib.util.spec_from_file_location('rd',W/'source/cylinder_step3_shock_layer_metrics.py');rd=importlib.util.module_from_spec(spec);spec.loader.exec_module(rd);plt.rcParams.update({'font.size':9,'axes.labelsize':10,'axes.titlesize':10,'axes.titleweight':'normal','xtick.labelsize':9,'ytick.labelsize':9,'figure.titlesize':12})
tr=ast.parse((W/'pod_all_kn_v14.py').read_text());exec(compile(ast.Module(body=[n for n in tr.body if isinstance(n,ast.FunctionDef) and n.name in ['smooth','outer','masked_triangulation','resample']],type_ignores=[]),'functions','exec'))
rows=pd.read_csv(W/'pod_all_kn_v12/SELECTED_EXACT_KN.csv').to_dict('records');rows += [r for r in pd.read_csv(W/'selected_cases.csv').to_dict('records') if r['sweep']=='Mach'];records=[];banks={};provenance=[];captions=[]
colors={'rho':'#2369ad','mach_proxy':'#e69f00','p':'#21834a','ttr':'#d55e00','trot':'#9467bd'};labels={'rho':r'$\rho$','mach_proxy':r'$M_{tr}$','p':r'$p$','ttr':r'$T_{tr}$','trot':r'$T_{rot}$'}
for row in rows:
 gas=row['gas'];sweep=row['sweep'];parameter=float(row['parameter']);d=rd.read_case(row);tri=masked_triangulation(d);v=np.array([np.cos(np.deg2rad(170)),np.sin(np.deg2rad(170))]);origin=np.array([R,0])+R*v;limits=[]
 for k,nm in enumerate(['x','y']):
  end=d[nm].max() if v[k]>0 else d[nm].min();t=(end-origin[k])/v[k]
  if t>0:limits.append(t)
 s=np.linspace(0,min(limits),2001);xy=origin[:,None]+v[:,None]*s;mass=6.63e-26 if gas=='Ar' else 4.65e-26;gamma=5/3 if gas=='Ar' else 1.4
 # Mach data have no imposed-state metadata here: use a measured upstream level only for marker diagnostics, explicitly recorded.
 vals={}
 for var in ['rho','ttr','trot','u','p']:
  if var not in d or np.nanmax(abs(d[var]))==0:continue
  ip=mtri.LinearTriInterpolator(tri,d[var]);vals[var]=np.asarray(ip(xy[0],xy[1]).filled(np.nan));vals[var][(xy[0]-R)**2+xy[1]**2<=R*R]=np.nan
 vals['mach_proxy']=np.abs(vals['u'])/np.sqrt(gamma*1.380649e-23*np.maximum(vals['ttr'],1e-20)/mass)
 tail=(s>s[-1]-.035)&(s<s[-1]-.005);rhoinf=float(row['nrho_in'])*mass if sweep=='Kn' else float(np.nanmedian(vals['rho'][tail]));q=vals['rho']/rhoinf;case_lambda=float(row['kn'])*2*R;spacing=np.sqrt(np.ptp(d['x'])*np.ptp(d['y'])/len(d['x']));sigma=max(.5*spacing,.01*case_lambda);marker=outer(s,q,sigma);rec={'gas':gas,'sweep':sweep,'parameter':parameter,'marker':marker,'rho_upstream_source':'imposed input' if sweep=='Kn' else 'measured farthest 5–35mm window; diagnostic only','rhoinf':rhoinf,'variables':{}}
 if marker:
  center=marker['s'];down=marker['downstream_s'];end=min(s[-1]-.005,center+max(4*marker['delta'],8*sigma));up=(s>=end-max(3*sigma,.005))&(s<=end);dw=abs(s-down)<max(sigma,.001524);interval=(s>=down)&(s<=end);xi=(s-center)/marker['delta'];reg={}
  for var,z in vals.items():
   qup=float(np.nanmedian(z[up]));qdn=float(np.nanmedian(z[dw]));jump=qdn-qup;sm=smooth(z,s[1]-s[0],sigma);grad=np.abs(np.gradient(sm,s));good=interval&np.isfinite(grad);peak=float(np.max(grad[good])) if good.any() else 0;spread=float(np.nanstd(z[up]));supported=bool(np.isfinite(jump) and abs(jump)>max(5*spread,1e-12) and peak>0)
   rec['variables'][var]={'up':qup,'down':qdn,'signed_jump':jump,'peak_abs_gradient':peak,'delta_m':abs(jump)/peak if supported else None,'normalization_supported':supported}
   reg[var]=(z-qup)/jump if supported else np.full(len(s),np.nan)
  banks[gas,sweep,parameter]=(s,vals,xi,reg)
 records.append(rec);provenance.append({'file':row['path'],'sha256':hashlib.sha256(Path(row['path']).read_bytes()).hexdigest(),'gas':gas,'sweep':sweep,'parameter':parameter});print('EXTRACTED',gas,sweep,parameter, bool(marker),flush=True)
def save(fig,name,cap):
 for ax in fig.axes:
  if ax.get_xscale()=='log':
   ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
   xx=sorted(set(float(v) for line in ax.lines for v in line.get_xdata()))
   if len(xx)<=3 and xx:ax.set_xticks(xx);ax.set_xticklabels([f'{v:g}' for v in xx])
 fig.tight_layout();fig.savefig(O/(name+'.pdf'),bbox_inches='tight');fig.savefig(O/(name+'.png'),dpi=140,bbox_inches='tight');plt.close(fig);captions.append({'file':name,'caption':cap})
for sweep in ['Kn','Mach']:
 fig,axs=plt.subplots(1,2,figsize=(10,4))
 for gas in ['Ar','N2']:
  rr=sorted([r for r in records if r['gas']==gas and r['sweep']==sweep and r['marker']],key=lambda r:r['parameter']);x=[r['parameter'] for r in rr]
  axs[0].plot(x,[r['marker']['s']*1000 for r in rr],'o' if len(x)<=2 else 'o-',label=gas);axs[1].plot(x,[r['marker']['delta']*1000 for r in rr],'o' if len(x)<=2 else 'o-',label=gas)
 for ax in axs:ax.set_xlabel(sweep);ax.legend();ax.grid(alpha=.2);ax.set_xscale('log' if sweep=='Kn' else 'linear')
 axs[0].set_ylabel('Density marker distance (mm)');axs[1].set_ylabel('Density gradient thickness (mm)');fig.suptitle('170° ray | supported outer compression only | archived-field diagnostic')
 save(fig,'Step3_'+sweep+'_Ar_N2_density_metrics','Re-extracted external density-compression markers on 170-degree ray; unsupported cases omitted and enumerated in EXTRACTION.json. Exported field spacing is not collision-cell resolution. Kn uses imposed density; Mach upstream density is measured, not verified imposed input. Diagnostic pending plateau audit; no uncertainty or convergence inferred.')
 for gas in ['Ar','N2']:
  rr=sorted([r for r in records if r['gas']==gas and r['sweep']==sweep and r['marker']],key=lambda r:r['parameter']);fig,axs=plt.subplots(1,3,figsize=(13,4))
  for var in colors:
   valid=[r for r in rr if r['variables'].get(var,{}).get('delta_m') is not None]
   if not valid:continue
   x=[r['parameter'] for r in valid]
   for ax,key in zip(axs,['delta_m','signed_jump','peak_abs_gradient']):ax.plot(x,[r['variables'][var][key] for r in valid],'o' if len(x)<=2 else 'o-',color=colors[var],label=labels[var]);ax.set_xlabel(sweep);ax.set_xscale('log' if sweep=='Kn' else 'linear');ax.grid(alpha=.2)
  for ax,lab in zip(axs,['Gradient thickness (m)','Signed jump (variable units)','Peak gradient (variable units/m)']):ax.set_ylabel(lab);ax.legend(fontsize=7)
  fig.suptitle(gas+' | outer density-defined interval | plateau-sensitive diagnostic')
  save(fig,'Step3_'+gas+'_'+sweep+'_variable_scales','Fixed variable colors; dimensional jumps and gradient maxima displayed separately. Same density-defined downstream/upstream windows used for all fields. Small jumps versus measured upstream variation excluded. Thermal overshoot may invalidate a plateau interpretation; these are diagnostics, not finalized relaxation lengths.')
  if gas=='N2':
   fig,axs=plt.subplots(2,3,figsize=(13,7))
   for j,r in enumerate(rr):
    s,vals,xi,reg=banks[gas,sweep,r['parameter']];col=plt.cm.viridis(j/max(1,len(rr)-1))
    for i,var in enumerate(['rho','ttr','trot']):
     axs[0,i].plot(s*1000,vals[var],color=col,label=f'{sweep}={r["parameter"]:g}');axs[1,i].plot(xi,reg.get(var,np.full(len(s),np.nan)),color=col)
   for i,var in enumerate(['rho','ttr','trot']):
    axs[0,i].set(xlabel='Distance outward (mm)',ylabel=labels[var]+' (dimensional)',xlim=(0,180));axs[1,i].set(xlabel=r'$(s-s_\rho)/\delta_\rho$',ylabel=labels[var]+r'$^*$',xlim=(-1,3));axs[1,i].annotate('downstream ←  → upstream',xy=(.05,.03),xycoords='axes fraction',fontsize=7)
   axs[0,0].legend(fontsize=6,ncol=2);fig.suptitle('N2 | dimensional companion + density-registered profiles | diagnostic')
   save(fig,'Step2c_N2_'+sweep+'_rho_registered_profile_collapse','Dimensional companion and signed jump-normalized registered profiles; no clipping of negative values. Only supported outer density markers shown. Original N2 Kn=.075 absent; no silent solver substitution. Plateau/jump diagnostics saved per case. Ray170 is not stagnation plane; different from legacy sector medians.')
(O/'EXTRACTION.json').write_text(json.dumps(records,indent=2));(O/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2));(O/'CAPTIONS.json').write_text(json.dumps(captions,indent=2));print('DONE',len(captions),flush=True)
