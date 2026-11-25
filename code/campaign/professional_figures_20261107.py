from pathlib import Path
import os,sys,json,hashlib,datetime,ast,csv,importlib.util
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
W=Path('campaign');O=W/'editorial_current_revision';F=O/'manuscript';E=O/'evidence'
old=O/'restored_rgb_figures.py';s=old.read_text()
baseline_source_sha=hashlib.sha256(old.read_bytes()).hexdigest()
# Preserve the established parser, finite masks, normalization and cached scalar data.
s=s[:s.index('report=dict(updated_utc=')]
s=s.replace("colors={'rho':'#1f77b4','mach_proxy':'#ff7f0e','p':'#2ca02c','ttr':'#d62728','trot':'#9467bd','u':'#606060'}","colors=dict.fromkeys(['rho','mach_proxy','p','ttr','trot','u'],'black')")
insert=r"""
from matplotlib.colors import ListedColormap,BoundaryNorm
from cycler import cycler
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':11,'axes.labelsize':11.5,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10.5,'axes.linewidth':.75,'lines.linewidth':1.3,'axes.prop_cycle':cycler(color=['black']),'savefig.dpi':300,'pdf.fonttype':42})
DENSITY_LEVELS=[.4,.8,1.2,1.6,2,2.5,3,4,5,6,7,8]
DENSITY_COLORS=['#173b91','#2166cc','#21a6db','#20cebf','#41db74','#a5e537','#e8e526','#ffb52b','#f77721','#da3d16','#9b160b']
DENSITY_CMAP=ListedColormap(DENSITY_COLORS)
STYLES=['-','--','-.',':',(0,(6,1.6,1.2,1.6)),(0,(3,1.5))]
MARKERS=['o','s','^','D','v','+']
"""
s=s.replace('manifest=[];hashcache={}',insert+'\nmanifest=[];hashcache={}')
a=s.index('def line(');b=s.index('def axesclean',a)
s=s[:a]+r"""def line(ax,x,y,color,label=None,style='o'):
 x=np.asarray(x);y=np.asarray(y);good=np.isfinite(x)&np.isfinite(y)
 order=list(labels.values());idx=order.index(label) if label in order else (1 if label=='N2' else 0)
 ax.plot(x[good],y[good],ls=STYLES[idx] if '-' in style else 'None',color='black',label=label,marker=MARKERS[idx],mfc='white',mec='black',ms=3.5,mew=.7)
 
"""+s[b:]
s=s.replace("levels=np.linspace(.4,8,39),cmap='turbo',extend='both'","levels=DENSITY_LEVELS,cmap=DENSITY_CMAP,norm=BoundaryNorm(DENSITY_LEVELS,len(DENSITY_COLORS)),extend='both'")
s=s.replace("fig.savefig(F/(name+'.pdf'),bbox_inches='tight',pad_inches=.10)","fig.savefig(F/(name+'.pdf'),dpi=600,bbox_inches='tight',pad_inches=.06)")
s=s.replace("dpi=165,bbox_inches='tight',pad_inches=.10","dpi=300,bbox_inches='tight',pad_inches=.06")
# Individual RGB contour assets replace compound bitmaps in the TeX layout.
a=s.index(' fig,axs=plt.subplots(2,2,figsize=(10,7.1)');b=s.index(' for mach in ',a);s=s[:a]+s[b:]
s=s.replace('figsize=(6.8,4.2)','figsize=(4.4,2.8)')
s=s.replace("for sweep in ['Kn','Mach']:","for sweep in ['Mach']:",1)
s=s.replace("for gas,sweep in [('Ar','Kn'),('Ar','Mach'),('N2','Kn'),('N2','Mach')]:","for gas,sweep in [('Ar','Mach'),('N2','Mach')]:").replace('figsize=(10.8,12)','figsize=(6.5,7.5)').replace('fontsize=11.5','fontsize=9.5')
a=s.index("fig,axs=plt.subplots(1,3,figsize=(11.2,3.6)");b=s.index("S=json.loads",a);s=s[:a]+s[b:]
s=s.replace('figsize=(9.8,3.7)','figsize=(6.2,2.9)').replace('figsize=(6.4,4.3)','figsize=(3.25,2.7)').replace('figsize=(10.4,7.0)','figsize=(6.3,4.8)').replace('figsize=(9,9.5)','figsize=(6.3,5.6)').replace("frameon=False,fontsize=14","frameon=False,fontsize=10")
a=s.index('# Thermal mode maps:');b=s.index('# Parameter coefficients:',a);s=s[:a]+s[b:]
s=s.replace("spec.loader.exec_module(rd)","spec.loader.exec_module(rd);plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':11,'axes.labelsize':11.5,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10.5})")
s=s.replace("fig.canvas.draw()","[t.set_fontfamily('STIXGeneral') for t in fig.findobj(matplotlib.text.Text)];[t.set_math_fontfamily('stix') for t in fig.findobj(matplotlib.text.Text)];fig.canvas.draw()")
s=s.replace("MaxNLocator(nbins=5)","MaxNLocator(nbins=3)")
s=s.replace("'Outer density location (mm)'","'Outer density\\nlocation (mm)'").replace("'Local density width (mm)'","'Local density\\nwidth (mm)'")
s=s.replace("'Cumulative scalar variance'","'Cumulative scalar\\nvariance fraction'").replace("'Weighted reconstruction error (%)'","'Weighted relative\\nreconstruction error (%)'")
s=s.replace("'Median projection error (%)','Median interpolation error (%)'","'Median projection\\nerror (%)','Median interpolation\\nerror (%)'")
s=s.replace("'Wall distance $s$ (mm)'","'$s$ (mm)'")
s=s.replace("aa.set_ylabel((r'$M_{x,\\rm tr}$' if v=='mach_proxy' else labels[v])+'\\n'+lab,fontsize=9.5)","aa.set_ylabel((r'$\\delta_{'+v+r'}$ (mm)' if j==0 else r'$\\Delta '+v+r'$' if j==1 else r'$G_{'+v+r'}$')+'\\n'+('' if j==0 else '('+(units[v] if j==1 else gradunits[v])+')'),fontsize=9.5)")
# Compact mathematical scalar symbols; avoid long vertical prose labels and crowded decimal Mach ticks.
s=s.replace("units={'rho':", "varsymbol={'rho':r'\\rho','mach_proxy':r'M_{x,\\rm tr}','p':'p','ttr':r'T_{\\rm tr}','u':r'u_x','trot':r'T_{\\rm rot}'}\nunits={'rho':",1)
s=s.replace("r'$\\delta_{'+v+r'}$ (mm)'","r'$\\delta_{'+varsymbol[v]+r'}$ (mm)'").replace("r'$\\Delta '+v+r'$'","r'$\\Delta '+varsymbol[v]+r'$'").replace("r'$G_{'+v+r'}$'","r'$G_{'+varsymbol[v]+r'}$'")
s=s.replace("for aa in axs[-1]:aa.set_xlabel(r'$Kn_\\infty$' if sweep=='Kn' else r'$M_\\infty$')","for aa in axs[-1]:aa.set_xlabel(r'$M_\\infty$');aa.set_xticks([5,10,15] if gas=='Ar' else [5,10,14])")
s=s.replace("save(fig,f'Step3_{gas}_{sweep}_variable_scales'","[ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,4),useMathText=True) for ax in axs.flat]\n save(fig,f'Step3_{gas}_{sweep}_variable_scales'")

exec(compile(s,str(old),'exec'),globals())
print('CACHED_PLOTS_DONE',len(manifest),flush=True)
# One physical contour per RGB asset, with vector axes and embedded serif fonts.
def rasterize(cf):
 for c in getattr(cf,'collections',[]):
  c.set_edgecolor('face');c.set_linewidth(.04);c.set_rasterized(True)
def contour_save(d,name,source,variable='rho',rhoinf=None,limits=None,label=None):
 with plt.rc_context({'font.size':14,'axes.labelsize':16,'xtick.labelsize':13.5,'ytick.labelsize':13.5}):
  fig,ax=plt.subplots(figsize=(4.4,2.8),layout='constrained')
  if variable=='rho':
   im=contours(d,ax,rhoinf);rasterize(im)
  else:
   tri=mtri.Triangulation(d['x'],d['y']);tt=tri.triangles
   x=d['x'];y=d['y'];mask=np.zeros(len(tt),bool)
   for a,b in [(0,1),(1,2),(2,0)]:
    px=x[tt[:,a]]-R;py=y[tt[:,a]];vx=x[tt[:,b]]-x[tt[:,a]];vy=y[tt[:,b]]-y[tt[:,a]];u=np.clip(-(px*vx+py*vy)/np.maximum(vx*vx+vy*vy,1e-30),0,1);mask|=(px+u*vx)**2+(py+u*vy)**2<R*R
   tri.set_mask(mask)
   finite=np.isfinite(d[variable]);assert finite.all()
   hi=float(np.max(d[variable]));levels=np.linspace(0,hi,25)
   im=ax.tricontourf(tri,d[variable],levels=levels,cmap='turbo',extend='max');rasterize(im)
   ax.add_patch(plt.Circle((R,0),R,fc='white',ec='black',lw=.8));ax.set_aspect('equal');ax.set_xlabel('$x$ (m)');ax.set_ylabel('$y$ (m)');ax.tick_params(direction='in',top=True,right=True)
  ax.set_xlim(float(np.min(d['x'])),float(np.max(d['x'])) if limits is None else limits[1]);ax.set_ylim(0,float(np.max(d['y'])) if limits is None else limits[3])
  cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.025);cb.set_label(label or r'$\rho/\rho_\infty$')
  if variable=='rho':cb.set_ticks([1,2,4,6,8])
  save(fig,name,source,'Original physical samples and finite domain; RGB data layer with vector serif axes. Density blue bin0.8–1.2 leaves values unchanged; no display smoothing/imputation.',variable=variable)
for gas in ['Ar','N2']:
 for kn in [.01,.025,.1,1.]:
  row=next(r for r in knrows if r['gas']==gas and abs(float(r['kn'])-kn)<1e-9);row=dict(row,nrho_in=float(row['nrho_in']),kn=kn)
  d=rd.read_case(row);rhoinf=row['nrho_in']*(6.63e-26 if gas=='Ar' else 4.65e-26)
  contour_save(d,f'Step1_kn_rho_{gas}_Kn{str(kn).replace(".","p")}',[Path(row['path'])],rhoinf=rhoinf)
  if gas=='N2' and kn==.01:
   tv=d.get('tvib');print('ORIGINAL_TVIB',None if tv is None else [float(np.min(tv)),float(np.max(tv))],flush=True)
   for var,lab in [('ttr',r'$T_{\rm tr}$ (K)'),('trot',r'$T_{\rm rot}$ (K)')]:
    contour_save(d,'Original_N2_M10_'+var,[Path(row['path'])],variable=var,label=lab)
# Figure2 exact original SWD/ray coordinates, matched by SHA to submitted raster.
B=Path('DS2V_source/Shock_Compare_Akhlaghi_v22_fixed_biglegend')
original_png=B/'checks_compare/compare_Ar_Kn0p01_M10p0_mach_sweep_argon.png'
assert sha(original_png)=='ad987eb74e828bb7ee8c9c0e21a227180bf4afc32edc59f562bbe3a03cc2d5c0'
points=list(csv.DictReader((B/'shock_compare_points.csv').open()))
points=[r for r in points if r['gas']=='Ar' and float(r['Kn'])==.01 and float(r['Mach'])==10]
r=points[0];flow=B.parent/r['flow_file'];assert flow.exists(),flow
row=next(r for r in machrows if r['gas']=='Ar' and float(r['mach'])==10);row=dict(row,path=str(flow));d=rd.read_case(row)
norm=float(np.median(d['rho'][d['x']<=np.quantile(d['x'],.08)]))
with plt.rc_context({'font.size':11,'axes.labelsize':12,'xtick.labelsize':10.5,'ytick.labelsize':10.5,'legend.fontsize':10}):
 fig,ax=plt.subplots(figsize=(5.2,3.15),layout='constrained');im=contours(d,ax,norm);rasterize(im)
 for method,color,marker,ls,lab in [('ray_center','black','o','-',r'Ray ridge'),('akhlaghi_sigma1','#9d248f','o','None',r'SWD, $\sigma=1$'),('akhlaghi_sigma0.1','#00a8bf','s','None',r'SWD, $\sigma=0.1$')]:
  rr=[p for p in points if p['method']==method or (method=='akhlaghi_sigma0.1' and 'sigma0.1' in p['method'])]
  rr=sorted(rr,key=lambda p:float(p['y']));assert rr,method
  ax.plot([float(p['x']) for p in rr],[float(p['y']) for p in rr],color=color,ls=ls,marker=marker,mfc='white' if method=='ray_center' else color,mec='black',mew=.35,ms=2.6,lw=1,label=lab)
 ax.set_xlim(float(d['x'].min()),float(d['x'].max()));ax.set_ylim(0,float(d['y'].max()));ax.legend(loc='upper right',frameon=True,facecolor='white',framealpha=1,edgecolor='.8')
 cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.025);cb.set_label(r'$\rho/\rho_{\infty,\rm measured}$');cb.set_ticks([1,2,4,6,8])
 save(fig,'compare_Ar_Kn0p01_M10p0_mach_sweep_argon',[flow,B/'shock_compare_points.csv',original_png],'Exact original566 marker rows; original sourcePNG SHA matched; no detector rerun, fit or new marker. Density normalization same original leftmost8% estimator.')
(E/'ORIGINAL_SHOCK_COMPARISON_POINTS.csv').write_bytes((B/'shock_compare_points.csv').read_bytes())
# Physical profiles: same cached numbers, larger independently readable panels.
def profile_figure(name,variables,curves,source,xlabel,xlim):
 n=len(variables);shape=(3,1) if n==3 else (2,2);fig,axs=plt.subplots(*shape,figsize=(4.9,6.05) if n==3 else (6.25,5.2),layout='constrained')
 axs=np.asarray(axs).ravel()
 for i,(key,lab) in enumerate(variables):
  ax=axs[i]
  for k,(tag,x,y) in enumerate(curves[key]):ax.plot(x,y,color='black',ls=['-','--','-.'][k],label=tag,lw=1.3)
  ax.set(xlabel='',ylabel=lab,xlim=xlim);axesclean(ax)
  visible=np.concatenate([np.asarray(y)[np.isfinite(y)&(np.asarray(x)>=xlim[0])&(np.asarray(x)<=xlim[1])] for tag,x,y in curves[key]])
  lo,hi=float(visible.min()),float(visible.max());pad=.06*max(hi-lo,abs(hi)*1e-4,1e-9);ax.set_ylim(lo-pad,hi+pad)
  if i==0:ax.legend(loc='best',frameon=False,fontsize=9)
 fig.supxlabel(xlabel,fontsize=11.5)
 save(fig,name,source,'Unchanged cached physical profiles and support, black engineering line styles; font/geometry only.')
# Heldout curves have already been independently reproduced and cached; never retrain here.
z=np.load(E/'HELDOUT_PREDICTION_PROFILES.npz')
for gas in ['Ar','N2']:
 variables=[('rho',r'$\rho/\rho_\infty$'),('ttr',r'$T_{\rm tr}/T_\infty$'),('u',r'$u_x/U_\infty$')]+([('trot',r'$T_{\rm rot}/T_\infty$')] if gas=='N2' else [])
 curves={k:[('Held-out DS2V',z['s_m']*1000,z[f'{gas}_{k}_true']),('Rank-two interpolation',z['s_m']*1000,z[f'{gas}_{k}_prediction'])] for k,_ in variables}
 profile_figure(f'figures/POD_heldout_{gas}_M10_Kn01',variables,curves,[E/'HELDOUT_PREDICTION_PROFILES.npz'],'Wall distance $s$ (mm)',(15,160))
# Planar/cylinder curves reproduce the exact accepted coordinate transformations.
A=W/'analysis_planar_v22_20261031';rep=json.loads((A/'REPORT.json').read_text())
scope={'np':np};tree=ast.parse((W/'nitrogen_validation_v21_20261031/analyze_planar_v22.py').read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gaussian_filter1d'],type_ignores=[]),'smooth','exec'),scope)
for gas in ['Ar','N2']:
 p=A/f'{gas}_planar_long.npz';zp=np.load(p);pc=rep['cases'][gas+'_planar_long']['config'];px=zp['x'];pm=zp['mean']
 g=np.gradient(scope['gaussian_filter1d'](pm[:,0],.0015/np.median(np.diff(px))),px);allowed=abs(px-rep['cases'][gas+'_planar_long']['center_m'])<.025;ii=np.where(allowed)[0][np.argmax(g[allowed])];center=px[ii]
 root=Path(json.loads((W/f'convergence_controls_v20/CASE_RESULT_C4_{gas}_L4.json').read_text())['path']);rr=json.loads((root/'REPORT.json').read_text())['cases'][f'C4_{gas}_L4'];cp=root/f'C4_{gas}_L4_ray170.npz';zc=np.load(cp);ss=zc['s'];marker=rr['profiles']['170']['mean_profile_marker']['standoff_m'];order=np.argsort(marker-ss)
 variables=[('rho',r'$\rho/\rho_\infty$'),('u',r'$u_x$ (m s$^{-1}$)'),('ttr',r'$T_{\rm tr}$ (K)')]+([('trot',r'$T_{\rm rot}$ (K)')] if gas=='N2' else [])
 curves={}
 for j,(key,lab) in enumerate(variables):
  py=pm[:,j]/(pc['nrho']*(6.63e-26 if gas=='Ar' else 4.65e-26)) if key=='rho' else pm[:,j];ci={'rho':0,'u':2,'ttr':1,'trot':3}[key]
  curves[key]=[('Planar, long domain',(px-center)*1000,py),('Cylinder L4',(marker-ss[order])*1000,zc['mean'][order,ci])]
 profile_figure(f'figures/Planar_cylinder_{gas}',variables,curves,[p,cp,root/'REPORT.json',A/'REPORT.json'],'Downstream distance (mm)',(-50,40))
# Restore independent validation with monochrome, clearly distinct controls.
C=W/'nitrogen_validation_v21_20261031';A=C/'analysis';camp=json.loads((C/'CAMPAIGN.json').read_text());rep=json.loads((A/'REPORT.json').read_text());ref=np.genfromtxt(A/'Alsmeyer_N2_M153_digitized_v21.csv',delimiter=',',names=True)
fig,ax=plt.subplots(figsize=(6.2,3.55),layout='constrained');rx=ref['x_over_lambda']-np.interp(.5,ref['rho_star'],ref['x_over_lambda']);ax.plot(rx,ref['rho_star'],'s',color='black',mfc='white',ms=3.5,mew=.7,label='Alsmeyer (digitized)')
ins=[C/'CAMPAIGN.json',A/'REPORT.json',A/'Alsmeyer_N2_M153_digitized_v21.csv']
for i,c in enumerate(camp['configs']):
 p=A/(c['name']+'.npz');ins.append(p);z=np.load(p);r1=c['nrho']*camp['models']['baseline']['mass'];ax.plot((z['x']-rep['cases'][c['name']]['center_m'])/c['transport']['lambda_viscosity_m'],(z['mean'][:,0]-r1)/(r1*(c['rankine_hugoniot']['compression']-1)),color='black',ls=STYLES[i],label=c['name'].rsplit('_',1)[-1].capitalize())
ax.set(xlim=(-8,8),xlabel=r'$(x-x_{\rho,1/2})/\lambda_{\mu,1}$',ylabel=r'$(\rho-\rho_1)/(\rho_2-\rho_1)$');axesclean(ax);ax.legend(frameon=False,fontsize=9.5);save(fig,'figures/N2_validation_density',ins,'Same original experimental digitization and frozen base/fine/long normalization; monochrome only.')
# Six independent RGB POD map assets retain the exact sparse mask.
for v in ['ttr','trot']:
 p=pathfor('N2',v,'Kn');z=np.load(p);ys=np.linspace(100,180,65);xi=np.linspace(-1,3,301);support=z['support'].astype(bool).reshape(65,301);valid=np.where(support.any(axis=1))[0]
 for j,value in enumerate([z['mean'],z['modes'][0],z['modes'][1]]):
  a=np.full(19565,np.nan);a[z['support'].astype(bool)]=value;a=a.reshape(65,301)
  fig,ax=plt.subplots(figsize=(2.25,2.35),layout='constrained')
  kw={} if j==0 else {'norm':TwoSlopeNorm(0,-float(np.nanmax(abs(a))),float(np.nanmax(abs(a))))}
  im=ax.pcolormesh(xi,ys,np.ma.masked_invalid(a),shading='auto',cmap='turbo' if j==0 else 'coolwarm',rasterized=True,**kw)
  ax.set(xlabel=r'$\xi_\rho$',ylabel=r'$\theta$ (deg)',ylim=(ys[valid.min()]-.625,ys[valid.max()]+.625));ax.tick_params(direction='in',top=True,right=True)
  cb=fig.colorbar(im,ax=ax,fraction=.06,pad=.03);cb.set_label((normlabels[v] if j==0 else rf'$\phi_{j}$'),fontsize=9)
  save(fig,f'Step2b_N2_Kn_{v}_'+['mean','mode1','mode2'][j],[p],'Strict original finite578-feature mask; no fill, smoothing, support extension, sign clipping or mode rotation.')
# Tables preserve every finite scalar value from the sparse diagnostic panels.
names={'rho':r'$\rho$ (kg m$^{-3}$)','mach_proxy':r'$M_{x,\rm tr}$ (1)','p':r'$p$ (Pa)','ttr':r'$T_{\rm tr}$ (K)','u':r'$u_x$ (m s$^{-1}$)','trot':r'$T_{\rm rot}$ (K)'}
tablemeta=[]
for gas,sweep,label in [('Ar','Kn','tab:ar_variable_scales_kn'),('Ar','Mach','tab:ar_variable_scales_mach'),('N2','Mach','tab:n2_variable_scales_mach')]:
 vs=['rho','mach_proxy','p','ttr','u' if gas=='Ar' else 'trot'];rs=sorted([r for r in ex if r['gas']==gas and r['sweep']==sweep and r['marker']],key=lambda r:r['parameter']);lines=[]
 for r in rs:
  for v in vs:
   q=r['variables'].get(v,{})
   if not q.get('normalization_supported'):continue
   lines.append([f"{r['parameter']:g}",names[v],f"{q['delta_m']*1000:.3f}",f"{q['signed_jump']:.4g}",f"{q['peak_abs_gradient']:.4g}"])
 tex=r'\begin{table}'+'\n'+r'\centering\small'+'\n'+r'\caption{'+gas+' '+sweep+r'-sweep local-anchor diagnostics on the $170^\circ$ ray. Widths are in mm; signed differences use the scalar units in column two and gradients use those units per metre. Only supported original observations are listed; missing fronts and normalization failures are not filled. These are local-gradient diagnostics, not equilibrium relaxation lengths.}\label{'+label+'}\n'+r'\begin{tabular}{llrrr}\toprule'+'\n'+(r'$Kn_\infty$' if sweep=='Kn' else r'$M_\infty$')+r' & Scalar (units) & $\delta_q$ (mm) & $\Delta q$ & $G_q$\\\midrule'+'\n'+'\n'.join(' & '.join(row)+r'\\' for row in lines)+'\n'+r'\bottomrule\end{tabular}'+'\n'+r'\end{table}'+'\n'
 dest=F/f'{gas}_{sweep}_variable_metrics_table.tex';dest.write_text(tex);tablemeta.append(dict(file=dest.name,rows=len(lines),all_values_from=str(expath)))
report=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_renderer_sha256=baseline_source_sha,source_script_sha256=sha(__file__),scope='Author-requested professional typography/layout; existing scientific arrays, original marker rows, physical support and normalization preserved; no solver job or scientific threshold change',assets=manifest,tables=tablemeta,density_display=dict(levels=DENSITY_LEVELS,colors=DENSITY_COLORS,uniform_upstream_blue_interval=[.8,1.2],raw_density_values_unchanged=True),original_png_hash_match=True,temperature_model_figure_pending=True)
(O/'PROFESSIONAL_FIGURES_20261107.json').write_text(json.dumps(report,indent=2));print('PROFESSIONAL_ASSETS_DONE',len(manifest),tablemeta,flush=True)
