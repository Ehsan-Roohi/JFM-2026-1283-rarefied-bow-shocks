
from pathlib import Path
import csv,json,hashlib,importlib.util,ast,datetime,tarfile,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator,NullLocator
W=Path('campaign');O=W/'editorial_current_revision';F=O/'manuscript';P=W/'pod_all_kn_v12/result_v14_20261025T025404Z';M=O/'evidence/restored_mach_pod'
F.mkdir(exist_ok=True,parents=True);R=.1524
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral','DejaVu Serif'],'mathtext.fontset':'stix','font.size':13,'axes.labelsize':14,'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':11,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.8,'lines.linewidth':1.55,'savefig.facecolor':'white'})
colors={'rho':'#1f77b4','mach_proxy':'#ff7f0e','p':'#2ca02c','ttr':'#d62728','trot':'#9467bd','u':'#606060'}
labels={'rho':r'$\rho$','mach_proxy':r'$M_{\rm tr}$','p':r'$p$','ttr':r'$T_{\rm tr}$','trot':r'$T_{\rm rot}$','u':r'$u_x$'}
normlabels={'rho':r'$\rho/\rho_\infty$','ttr':r'$T_{\rm tr}/T_\infty$','trot':r'$T_{\rm rot}/T_\infty$','u':r'$u_x/U_\infty$'}
manifest=[];hashcache={}
def sha(p):
 p=Path(p)
 if str(p) not in hashcache:
  with p.open('rb') as f:hashcache[str(p)]=hashlib.file_digest(f,'sha256').hexdigest()
 return hashcache[str(p)]
def save(fig,name,inputs,caption,**meta):
 fig.canvas.draw()
 fig.savefig(F/(name+'.pdf'),bbox_inches='tight',pad_inches=.10)
 fig.savefig(F/(name+'.png'),dpi=165,bbox_inches='tight',pad_inches=.10)
 plt.close(fig)
 manifest.append(dict(file=name,source=[dict(path=str(p),sha256=sha(p)) for p in dict.fromkeys(inputs)],caption=caption,outputs={s:sha(F/(name+s)) for s in ['.pdf','.png']},**meta))
def line(ax,x,y,color,label=None,style='o'):
 x=np.asarray(x);y=np.asarray(y);good=np.isfinite(x)&np.isfinite(y)
 ax.plot(x[good],y[good],style,color=color,label=label,ms=4)
def axesclean(ax):
 ax.grid(color='.87',lw=.5);ax.tick_params(direction='in',top=True,right=True);ax.xaxis.set_major_locator(MaxNLocator(nbins=5)) if ax.get_xscale()=='linear' else None
def render_loo():
 # Same cached cases, ranks and median definitions as the original renderer.
 # Typography alone changes; no solver, scientific analysis or scalar norm changes.
 rows=json.loads((P/'SUMMARY.json').read_text())
 rows=[r for r in rows if r['gas']=='N2' and r['basis']=='physical' and r.get('grid')==[480,320] and r.get('family')=='historical_only']
 curves=[]
 with plt.rc_context({'font.size':18,'axes.labelsize':18,'xtick.labelsize':18,'ytick.labelsize':18,'legend.fontsize':18}):
  fig,axs=plt.subplots(1,2,figsize=(10.5,4.8),layout='constrained')
  fig.get_layout_engine().set(h_pad=.15,w_pad=.08)
  for r in rows:
   loo=r['leave_one_out'];rs=sorted(set(x['r'] for x in loo))
   for j,key in enumerate(['conditional_projection_L2_percent','logKn_coefficient_interpolation_L2_percent']):
    x=[];y=[]
    for rank in rs:
     a=[z[key] for z in loo if z['r']==rank and key in z]
     if a:x.append(rank);y.append(float(np.median(a)))
    line(axs[j],x,y,colors[r['variable']],labels[r['variable']],'o-')
    curves.append(dict(variable=r['variable'],observable=key,ranks=x,median_L2_percent=y))
  for j,ax in enumerate(axs):
   ax.set_ylabel(['Median projection error (%)','Median interpolation error (%)'][j]);axesclean(ax);ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=5))
  fig.supxlabel('Retained physical POD modes',fontsize=18)
  handles,leglabels=axs[0].get_legend_handles_labels()
  fig.legend(handles,leglabels,loc='outside upper center',ncol=6,fontsize=18,frameon=True)
  save(fig,'LeaveOneOut_POD_reconstruction_errors',[P/'SUMMARY.json'],'Historical-onlyN2 physical480basis; leftconditional heldout projection(allcases), righttraining-onlylogKncoefficientinterpolation(interioronly). Differentheldout sets; medians, no fabricatedconfidenceband. Weighted fullfield relativeL2percent.',typography=dict(font_pt=18,legend_pt=18,shared_legend=True),curve_data_sha256=hashlib.sha256(json.dumps(curves,sort_keys=True,separators=(',',':')).encode()).hexdigest())
 return curves

def targeted_loo():
 old=json.loads((O/'RESTORED_RGB_FIGURES_MANIFEST.json').read_text())
 before={a['file']:dict(a['outputs']) for a in old['assets']}
 curves=render_loo();new=manifest[-1]
 matches=[i for i,a in enumerate(old['assets']) if a['file']==new['file']]
 assert len(matches)==1
 old['assets'][matches[0]]=new
 old['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();old['source_script_sha256']=sha(__file__)
 old['last_change']='Only LeaveOneOut_POD_reconstruction_errors typography: 18pt labels/ticks/shared legend. Same cached median curves and scalar normalization.'
 (O/'RESTORED_RGB_FIGURES_MANIFEST.json').write_text(json.dumps(old,indent=2))
 checked=[]
 for a in old['assets']:
  for ext,digest in a['outputs'].items():
   assert sha(F/(a['file']+ext))==digest,(a['file'],ext)
   checked.append(a['file']+ext)
  if a['file']!=new['file']:assert a['outputs']==before[a['file']]
 bundle=O/'restored_rgb_figures_transfer.tar.gz'
 with tarfile.open(bundle,'w:gz') as t:
  t.add(__file__,arcname='restored_rgb_figures.py');t.add(O/'RESTORED_RGB_FIGURES_MANIFEST.json',arcname='RESTORED_RGB_FIGURES_MANIFEST.json')
  for ext in ['.pdf','.png']:t.add(F/(new['file']+ext),arcname='manuscript/'+new['file']+ext)
 receipt=dict(path=str(bundle),bytes=bundle.stat().st_size,sha256=sha(bundle),assets=1,transfer_scope='Targeted typography update only; complete manifest verifies all 28 asset families / 56 files',files_transferred=2,all_files_verified=len(checked),unchanged_asset_families=27,source_script_sha256=sha(__file__),manifest_sha256=sha(O/'RESTORED_RGB_FIGURES_MANIFEST.json'),curve_data_sha256=new['curve_data_sha256'],outputs=new['outputs'])
 (O/'RESTORED_RGB_TRANSFER.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
 return receipt

if '--only-loo' in sys.argv:
 targeted_loo();raise SystemExit(0)

def contours(d,ax,rhoinf):
 t=mtri.Triangulation(d['x'],d['y']);tt=t.triangles;x=d['x'];y=d['y'];mask=np.zeros(len(tt),bool)
 for a,b in [(0,1),(1,2),(2,0)]:
  px=x[tt[:,a]]-R;py=y[tt[:,a]];vx=x[tt[:,b]]-x[tt[:,a]];vy=y[tt[:,b]]-y[tt[:,a]];u=np.clip(-(px*vx+py*vy)/np.maximum(vx*vx+vy*vy,1e-30),0,1);mask|=(px+u*vx)**2+(py+u*vy)**2<R*R
 t.set_mask(mask)
 im=ax.tricontourf(t,d['rho']/rhoinf,levels=np.linspace(.4,8,39),cmap='turbo',extend='both')
 ax.add_patch(plt.Circle((R,0),R,fc='white',ec='black',lw=1))
 ax.set_aspect('equal');ax.set_xlabel('$x$ (m)');ax.set_ylabel('$y$ (m)')
 ax.tick_params(direction='in',top=True,right=True)
 return im
spec=importlib.util.spec_from_file_location('reader',W/'source/cylinder_step3_shock_layer_metrics.py');rd=importlib.util.module_from_spec(spec);spec.loader.exec_module(rd)
knrows=list(csv.DictReader((W/'pod_all_kn_v12/SELECTED_EXACT_KN.csv').open()));machrows=[r for r in csv.DictReader((W/'selected_cases.csv').open()) if r['sweep']=='Mach' and float(r['mach']).is_integer()]
nom=json.loads((P/'INPUTS.json').read_text());refs={g:next(a['nominal_state']['rho'] for a in nom if a['gas']==g and a['kn']==.01) for g in ['Ar','N2']}
for gas in ['Ar','N2']:
 fig,axs=plt.subplots(2,2,figsize=(10,7.1),layout='constrained');inputs=[];used=[]
 for ax,kn in zip(axs.flat,[.01,.025,.1,1.]):
  row=next(r for r in knrows if r['gas']==gas and abs(float(r['kn'])-kn)<1e-9);row=dict(row,nrho_in=float(row['nrho_in']),kn=kn)
  d=rd.read_case(row);norm=row['nrho_in']*(6.63e-26 if gas=='Ar' else 4.65e-26);im=contours(d,ax,norm)
  ax.set_xlim(float(d['x'].min()),float(d['x'].max()));ax.set_ylim(0,float(d['y'].max()));inputs.append(Path(row['path']));used.append(dict(kn=kn,rhoinf=norm,limits=[ax.get_xlim(),ax.get_ylim()]))
 cb=fig.colorbar(im,ax=axs.ravel().tolist(),fraction=.035,pad=.02);cb.set_label(r'$\rho/\rho_\infty$');cb.set_ticks([1,2,4,6,8])
 save(fig,'Step1_kn_rho_'+gas,inputs+[P/'INPUTS.json'],'Mach10; panels top-left0.01,top-right0.025,bottom-left0.1,bottom-right1. Original RGB contour identity retained; imposed m*n normalization, identical levels0.4â€“8 in both gases, black body boundary, honest per-case exported domains; no unverified ridge overlay.',panels=used)
 for mach in ([5,10,15] if gas=='Ar' else [5,10,14]):
  row=next(r for r in machrows if r['gas']==gas and float(r['mach'])==mach);d=rd.read_case(row);fig,ax=plt.subplots(figsize=(6.8,4.2),layout='constrained');im=contours(d,ax,refs[gas]);ax.set_xlim(float(d['x'].min()),min(.65,float(d['x'].max())));ax.set_ylim(0,min(.4,float(d['y'].max())));cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.025);cb.set_label(r'$\rho/\rho_{\infty,\rm nominal}$');cb.set_ticks([1,2,4,6,8])
  name='Step1_mach_rho_'+gas+('' if mach==10 else '_M'+str(mach))
  save(fig,name,[Path(row['path']),P/'INPUTS.json'],'One Mach number per RGB figure; original physical density field with baseline nominal Kn0.01 imposed-density scale. Case-wise Mach incoming states are not independently verified; no field/width/amplitude fitted.',gas=gas,mach=mach,nominal_rhoinf=refs[gas])
# Historical Kn density metrics use the one primary v14 receipt, at170degrees only.
markers=list(csv.DictReader((P/'MARKERS.csv').open()))
km=[r for r in markers if abs(float(r['angle'])-170)<1e-9 and float(r['smoothing_factor'])==1 and r['supported']=='True']
expath=W/'reviewer_figures_v4_20261031/EXTRACTION.json';ex=json.loads(expath.read_text())
def metricrows(sweep,gas):
 if sweep=='Kn':return sorted([dict(parameter=float(r['kn']),s=float(r['s']),delta=float(r['delta'])) for r in km if r['gas']==gas],key=lambda a:a['parameter'])
 return sorted([dict(parameter=r['parameter'],s=r['marker']['s'],delta=r['marker']['delta']) for r in ex if r['sweep']=='Mach' and r['gas']==gas and r['marker']],key=lambda a:a['parameter'])
for sweep in ['Kn','Mach']:
 fig,axs=plt.subplots(1,2,figsize=(9.8,3.7),layout='constrained')
 for gas,c in [('Ar','#1f77b4'),('N2','#d62728')]:
  rs=metricrows(sweep,gas);x=[r['parameter'] for r in rs]
  for ax,key in zip(axs,['s','delta']):line(ax,x,[r[key]*1000 for r in rs],c,gas);axesclean(ax);ax.set_xlabel(r'$Kn_\infty$' if sweep=='Kn' else r'$M_\infty$')
 for ax in axs:
  if sweep=='Kn':ax.set_xscale('log');ax.xaxis.set_minor_locator(NullLocator());ax.set_xticks([.01,.025]);ax.set_xticklabels(['0.01','0.025'])
  ax.legend()
 axs[0].set_ylabel('Outer density location (mm)');axs[1].set_ylabel('Local density width (mm)')
 save(fig,'Step3_'+sweep+'_Ar_N2_density_metrics',[P/'MARKERS.csv',expath],'Supported170degree outer markers only; individual points, no fitted or connecting trend. Kn primary v14 selection uses imposed upstream density; Mach v4 selection uses measured far-state diagnostic. Not angular medians or equilibrium widths.',observable='single170degree ray; not median overangles')
units={'rho':r'kg m$^{-3}$','p':'Pa','u':r'm s$^{-1}$','ttr':'K','trot':'K','mach_proxy':'1'}
gradunits={'rho':r'kg m$^{-4}$','p':r'Pa m$^{-1}$','u':r's$^{-1}$','ttr':r'K m$^{-1}$','trot':r'K m$^{-1}$','mach_proxy':r'm$^{-1}$'}
for gas,sweep in [('Ar','Kn'),('Ar','Mach'),('N2','Kn'),('N2','Mach')]:
 vs=['rho','mach_proxy','p','ttr','u' if gas=='Ar' else 'trot'];fig,axs=plt.subplots(5,3,figsize=(10.8,12),layout='constrained',sharex=True)
 for i,v in enumerate(vs):
  rr=sorted([r for r in ex if r['gas']==gas and r['sweep']==sweep and r['marker'] and r['variables'].get(v,{}).get('normalization_supported')],key=lambda r:r['parameter'])
  for j,(key,mul,lab) in enumerate([('delta_m',1000,'Gradient width (mm)'),('signed_jump',1,'Signed jump ('+units[v]+')'),('peak_abs_gradient',1,'Peak |gradient| ('+gradunits[v]+')')]):
   aa=axs[i,j];line(aa,[r['parameter'] for r in rr],[r['variables'][v][key]*mul for r in rr],colors[v]);aa.set_ylabel((r'$M_{x,\rm tr}$' if v=='mach_proxy' else labels[v])+'\n'+lab,fontsize=11.5);axesclean(aa)
   if sweep=='Kn':aa.set_xscale('log');aa.xaxis.set_minor_locator(NullLocator())
 for aa in axs[-1]:aa.set_xlabel(r'$Kn_\infty$' if sweep=='Kn' else r'$M_\infty$')
 if sweep=='Kn':
  for aa in axs[-1]:aa.set_xticks([.01,.025]);aa.set_xticklabels(['0.01','0.025'])
 save(fig,f'Step3_{gas}_{sweep}_variable_scales',[expath],'Five rows; per-variable gradient widths, signed local jumps and positive peak gradient magnitudes in physical units. The M_x,tr diagnostic is abs(ux)/sqrt(gamma*k*Ttr/m); it differs from velocity-magnitude M_tr used in POD. v4 diagnostic has its own2001point sampling/smoothing; it is distinct from primary density-marker receipt. Unsupported anchors omitted; points only, no fitted trend or universal thermal ordering.',row_variables=vs,plateau_audit='diagnostic only')
fig,axs=plt.subplots(1,3,figsize=(11.2,3.6),layout='constrained')
for gas,c in [('Ar','#1f77b4'),('N2','#d62728')]:
 for j,(sweep,key) in enumerate([('Kn','s'),('Kn','delta'),('Mach','s')]):
  rr=metricrows(sweep,gas);line(axs[j],[r['parameter'] for r in rr],[r[key]*1000 for r in rr],c,gas)
for j,ax in enumerate(axs):
 axesclean(ax);ax.legend();ax.set_ylabel(['Outer location (mm)','Local width (mm)','Outer location (mm)'][j]);ax.set_xlabel(r'$Kn_\infty$' if j<2 else r'$M_\infty$')
 if j<2:ax.set_xscale('log');ax.xaxis.set_minor_locator(NullLocator());ax.set_xticks([.01,.025]);ax.set_xticklabels(['0.01','0.025'])
save(fig,'JFM_scaling_three_panel_clean',[P/'MARKERS.csv',expath],'Observed supported marker points only. No empirical global fit, no highKn marker fabrication and no Hornung calibration. Firsttwo panels primaryKn170degree; lastMachv4diagnostic.')
S=json.loads((P/'SUMMARY.json').read_text());MS=json.loads((M/'SUMMARY.json').read_text())
def selected_rows(gas,sweep,basis):
 ss=S if sweep=='Kn' else MS
 return [r for r in ss if r['gas']==gas and r['basis']==basis and (sweep!='Kn' or basis!='density_attached' or r.get('smoothing_factor')==1.) and (sweep!='Kn' or basis!='physical' or (r.get('grid')==[480,320] and r.get('family')==('all_Kn_historical' if gas=='Ar' else 'historical_only')))]
def pathfor(gas,var,sweep):
 return (P/f'{gas}_{var}_attached_sigma1.0.npz') if sweep=='Kn' else M/f'{gas}_{var}_attached_Mach.npz'
def fullbank(z):
 support=z['support'].astype(bool);x=z['mean']+z['coefficients']@z['modes'];a=np.full((len(x),19565),np.nan);a[:,support]=x;return a.reshape(len(x),65,301),support.reshape(65,301)
# Four cumulative-spectrum and four reconstruction-error original assets.
for gas in ['Ar','N2']:
 for sweep in ['Kn','Mach']:
  rows=selected_rows(gas,sweep,'density_attached');fig,ax=plt.subplots(figsize=(6.4,4.3),layout='constrained')
  for r in rows:
   e=np.array(r['spectrum'])[:r['snapshots']-1];line(ax,np.arange(1,len(e)+1),np.cumsum(e),colors[r['variable']],labels[r['variable']],'o-')
  ax.set(xlabel='Number of POD modes',ylabel='Cumulative scalar variance',ylim=(0,1.025));axesclean(ax);ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=7));ax.legend(ncol=2)
  save(fig,f'POD_spectrum_{gas}_{sweep}_attached',[P/'SUMMARY.json' if sweep=='Kn' else M/'SUMMARY.json'],'Mean-subtracted scalar captured variance, not kinetic energy; actual density-attached cases and masks. MaximumrankNs-1.',gas=gas,sweep=sweep,snapshots=rows[0]['snapshots'],common_points=rows[0]['common_points'])
  fig,ax=plt.subplots(figsize=(6.4,4.3),layout='constrained')
  for r in rows:
   rr=[a for a in r['reconstruction'] if a['r']>0];line(ax,[a['r'] for a in rr],[a['relative_L2_percent'] for a in rr],colors[r['variable']],labels[r['variable']],'o-')
  ax.set(xlabel='Number of POD modes',ylabel='Weighted reconstruction error (%)',yscale='log');axesclean(ax);ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=7));ax.legend(ncol=2)
  save(fig,f'POD_reconstruction_error_{gas}_{sweep}_attached',[P/'SUMMARY.json' if sweep=='Kn' else M/'SUMMARY.json'],'In-sample weighted scalar reconstruction relative to full field, percent. Nearzero at full rank is algebraic reconstruction, not validation/prediction.')
# Registered Kn comparison keeps physical/attached2x3 design, using exactly the same cached finite supported samples.
vars3=['rho','ttr','trot'];ks=[]
z=np.load(pathfor('N2','rho','Kn'));_,supp=fullbank(z);ia=int(np.argmax(supp.sum(axis=1)));angle=float(np.linspace(100,180,65)[ia]);xi=np.linspace(-1,3,301)
fig,axs=plt.subplots(2,3,figsize=(10.4,7.0),layout='constrained');inputs=[]
for j,v in enumerate(vars3):
 p=pathfor('N2',v,'Kn');inputs.append(p);z=np.load(p);a,_=fullbank(z)
 for k,kn in enumerate(z['kn']):
  mr=next(r for r in markers if r['gas']=='N2' and float(r['kn'])==kn and float(r['smoothing_factor'])==1 and float(r['angle'])==angle and r['supported']=='True')
  ss=float(mr['s'])+xi*float(mr['delta']);c=plt.get_cmap('viridis')(.08+.82*k/max(1,len(z['kn'])-1))
  axs[0,j].plot(ss*1000,a[k,ia],color=c,label=f'Kn={kn:g}');axs[1,j].plot(xi,a[k,ia],color=c)
 for i in [0,1]:axs[i,j].set_ylabel(normlabels[v]);axesclean(axs[i,j])
 axs[0,j].set_xlabel('Wall distance $s$ (mm)');axs[1,j].set_xlabel(r'$\xi_\rho$ (upstream)')
axs[0,0].legend()
save(fig,'Step2c_N2_Kn_rho_registered_profile_collapse',inputs+[P/'MARKERS.csv'],'Four supported N2Kn cases; physical(top)and density-attached(bottom)coordinates of identical finite samples at selectedangle. Nominal scalar normalization, no independent shifts/amplitudefit/imputation; not equilibrium-jump collapse.',angle_deg=angle,kn=z['kn'].tolist())
# One Mach per registered physical/attached comparison; primary originalbasename isM10.
mmarkers=json.loads((M/'MARKERS.json').read_text());z=np.load(pathfor('N2','rho','Mach'));_,supp=fullbank(z);ia=int(np.argmax(supp.sum(axis=1)));angleM=float(np.linspace(100,180,65)[ia])
fig,axs=plt.subplots(2,3,figsize=(10.4,7.0),layout='constrained');inputs=[]
for j,v in enumerate(vars3):
 p=pathfor('N2',v,'Mach');inputs.append(p);z=np.load(p);a,_=fullbank(z);k=int(np.where(z['kn']==10)[0][0]);mr=next(r for r in mmarkers if r['gas']=='N2' and r['mach']==10 and r['angle']==angleM and r['supported']);ss=mr['s']+xi*mr['delta']
 for i,x in enumerate([ss*1000,xi]):
  axs[i,j].plot(x,a[k,ia],color=colors[v],label=labels[v]);axesclean(axs[i,j]);axs[i,j].set_ylabel(r'$\rho/\rho_{\rm ref}$' if v=='rho' else normlabels[v]);axs[i,j].set_xlabel('Wall distance $s$ (mm)' if i==0 else r'$\xi_\rho$ (upstream)')
save(fig,'Step2c_N2_Mach_rho_registered_profile_collapse',inputs+[M/'MARKERS.json',M/'METHOD.json'],'Single originalN2Mach10 field, physical(top)/density-attached(bottom)coordinates of identical finite samples. Thermal normalizationT/200K; density uses fixedsamegasdiagnosticreference ofMachanalysis, not verifiedcasewiseimposed state. No differentMachprofiles mixed.',mach=10,angle_deg=angleM)
# Thermal mode maps: honest sparse common support, cropped only to finite angular boundingbox.
fig,axs=plt.subplots(2,3,figsize=(10.5,6.5),layout='constrained');inputs=[];mapmeta=[]
for i,v in enumerate(['ttr','trot']):
 p=pathfor('N2',v,'Kn');inputs.append(p);z=np.load(p);a,sup=fullbank(z);ys=np.linspace(100,180,65);validrows=np.where(sup.any(axis=1))[0];lo,hi=validrows.min(),validrows.max();data=[z['mean'],z['modes'][0],z['modes'][1]]
 for j,d in enumerate(data):
  f=np.full(19565,np.nan);f[z['support'].astype(bool)]=d;f=f.reshape(65,301)
  if j==0:im=axs[i,j].pcolormesh(xi,ys,np.ma.masked_invalid(f),shading='auto',cmap='turbo')
  else:
   vmax=float(np.nanmax(abs(f)));im=axs[i,j].pcolormesh(xi,ys,np.ma.masked_invalid(f),shading='auto',cmap='coolwarm',norm=TwoSlopeNorm(0,-vmax,vmax))
  axs[i,j].set_ylim(ys[lo]-.625,ys[hi]+.625);axs[i,j].set_xlabel(r'$\xi_\rho$');axs[i,j].set_ylabel(r'$\theta$ (deg)');cb=fig.colorbar(im,ax=axs[i,j],fraction=.045,pad=.025);cb.set_label(normlabels[v]+' mean' if j==0 else labels[v]+r' mode '+str(j),fontsize=11)
 r=next(r for r in selected_rows('N2','Kn','density_attached') if r['variable']==v);mapmeta.append(dict(variable=v,E1=r['E1'],E2=r['spectrum'][1],angle_limits_deg=[float(ys[lo]-.625),float(ys[hi]+.625)],points=int(sup.sum())))
save(fig,'Step2b_N2_Kn_POD_modes_thermal_combined_JFM_v4_smoother',inputs+[P/'SUMMARY.json'],'RowsTtr,Trot; columnsmean,mode1,mode2. Sparse strict finite common support is unfilled. Croppedangularboundingbox is explicit; RGBmean/diverging signedmodes, no extrapolation or smoothingfill.',maps=mapmeta)
# Parameter coefficients: retain RGB mode traces but use audited four-case bank.
fig,axs=plt.subplots(3,2,figsize=(9,9.5),layout='constrained');inputs=[]
for ax,v in zip(axs.flat,['rho','mach_proxy','p','ttr','trot']):
 p=pathfor('N2',v,'Kn');inputs.append(p);z=np.load(p)
 for k in range(min(3,len(z['kn'])-1)):
  coeff=z['coefficients'][:,k];den=max(abs(coeff));ax.plot(z['kn'],coeff/den,'o'+['-','--',':'][k],label=rf'$a_{k+1}$',ms=4)
 ax.set_xscale('log');ax.xaxis.set_minor_locator(NullLocator());ax.set_xlabel(r'$Kn_\infty$');ax.set_ylabel(labels[v]+r': $a_k/\max_j|a_{jk}|$');axesclean(ax)
axs[-1,-1].axis('off')
handles,leglabels=axs[0,0].get_legend_handles_labels()
axs[-1,-1].legend(handles,leglabels,loc='center',ncol=1,frameon=False,fontsize=14)
save(fig,'POD_coefficients_vs_parameter',inputs,'Actual4-caseN2attached coefficients, eachmode dividedby ownmaximumabsolute trainingcoefficient. Mode signs retained; theseareparameter-indexed coefficients, not temporal evolution or predictions.')
# Held-out error diagnostics distinguish projection from actual interior interpolation.
render_loo()
report=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_script_sha256=sha(__file__),scope='Original graphical identity/RGB vector quality restored with audited data; no simulation, controller, physics or otherproject modification',assets=manifest,original_comparison_png_untouched=True)
(O/'RESTORED_RGB_FIGURES_MANIFEST.json').write_text(json.dumps(report,indent=2))
bundle=O/'restored_rgb_figures_transfer.tar.gz'
with tarfile.open(bundle,'w:gz') as t:
 t.add(__file__,arcname='restored_rgb_figures.py');t.add(O/'RESTORED_RGB_FIGURES_MANIFEST.json',arcname='RESTORED_RGB_FIGURES_MANIFEST.json')
 for a in manifest:
  for ext in ['.pdf','.png']:t.add(F/(a['file']+ext),arcname='manuscript/'+a['file']+ext)
receipt=dict(path=str(bundle),bytes=bundle.stat().st_size,sha256=sha(bundle),assets=len(manifest))
(O/'RESTORED_RGB_TRANSFER.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
