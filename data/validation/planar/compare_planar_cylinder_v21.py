"""Physical profiles: planar long-domain versus mature cylinder L4.
This comparison contains wall/deceleration/obliquity as well as curvature.
"""
from pathlib import Path
import json,hashlib,ast
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path('campaign')
O=W/'analysis_planar_v22_20261031';Q=W/'convergence_controls_v20'
scope={'np':np}
src=W/'nitrogen_validation_v21_20261031/analyze_planar_v22.py'
tr=ast.parse(src.read_text());exec(compile(ast.Module(body=[n for n in tr.body if isinstance(n,ast.FunctionDef) and n.name=='gaussian_filter1d'],type_ignores=[]),'gaussian','exec'),scope)
smooth=scope['gaussian_filter1d'];report=json.loads((O/'REPORT.json').read_text());prov=[]
plt.rcParams.update({'font.size':10,'figure.titlesize':12})
for gas in ['Ar','N2']:
 pd=O/(gas+'_planar_long.npz');p=np.load(pd);pc=report['cases'][gas+'_planar_long']['config'];mass=6.63e-26 if gas=='Ar' else 4.65e-26
 px=p['x'];pm=p['mean'];sigma=.0015
 grad=np.gradient(smooth(pm[:,0],sigma/np.median(np.diff(px))),px)
 allowed=abs(px-report['cases'][gas+'_planar_long']['center_m'])<.025
 pi=np.where(allowed)[0][np.argmax(grad[allowed])];center=px[pi]
 root=Path(json.loads((Q/f'CASE_RESULT_C4_{gas}_L4.json').read_text())['path'])
 rr=json.loads((root/'REPORT.json').read_text())['cases'][f'C4_{gas}_L4']
 for angle in [170,180]:
  cp=root/f'C4_{gas}_L4_ray{angle}.npz';b=np.load(cp);s=b['s'];marker=rr['profiles'][str(angle)]['mean_profile_marker'];assert marker is not None
  assert p['B'].shape[0]==b['blocks'].shape[0]==10
  variables=['rho','u_x','Ttr']+(['Trot'] if gas=='N2' else [])
  fig,axs=plt.subplots(1,len(variables),figsize=(12,3.9))
  for j,var in enumerate(variables):
   # Bank order rho/rhoinf,Ttr,u_x,Trot,h/lambda,dt/tau,count.
   cy=b['mean'][:,{'rho':0,'Ttr':1,'u_x':2,'Trot':3}[var]]
   py=pm[:,j]/(pc['nrho']*mass) if var=='rho' else pm[:,j]
   axs[j].plot((px-center)*1000,py,c='#2369ad',lw=1.5,label='Planar, long domain')
   order=np.argsort(marker['standoff_m']-s)
   axs[j].plot((marker['standoff_m']-s[order])*1000,cy[order],c='#d55e00',lw=1.5,label='Cylinder L4')
   axs[j].set(xlabel='Distance downstream of density gradient (mm)',ylabel={'rho':r'$\rho/\rho_\infty$','u_x':r'$u_x$ (m/s)','Ttr':r'$T_{tr}$ (K)','Trot':r'$T_{rot}$ (K)'}[var],xlim=(-50,40));axs[j].grid(alpha=.2)
  axs[0].legend(fontsize=8)
  raylabel='170° ray' if angle==170 else '180° diagnostic: first gas-cell row, not exact particle plane'
  fig.suptitle(gas+' M=10, upstream200K | 10 blocks, 2–4ms | '+raylabel+'\nTranslation only; geometry, wall deceleration and numerical setups differ')
  fig.tight_layout();name=f'Supplement_{gas}_planar_cylinder_{angle}'
  for ext in ['png','pdf']:fig.savefig(O/(name+'.'+ext),dpi=170,bbox_inches='tight')
  plt.close(fig)
  prov.append({'file':name,'planar_bank':str(pd),'planar_sha256':hashlib.sha256(pd.read_bytes()).hexdigest(),
   'cylinder_bank':str(cp),'cylinder_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),
   'cylinder_archive_receipt':str(root/'ARCHIVE_VERIFIED.json'),'planar_center_gradient_m':float(center),
   'cylinder_marker_m':marker['standoff_m'],'smoothing_for_locations_m':sigma,
   'scope':'Dimensional field comparison, rho normalized by imposed mass density only. Planar x increases downstream; cylinder s increases upstream, so downstream coordinate is s_marker-s. u_x is streamwise velocity, not ray-normal velocity at170degrees. Not a curvature-only experiment, not external validation, not a matched numerical-discretization study.'})
(O/'PLANAR_CYLINDER_PROVENANCE.json').write_text(json.dumps(prov,indent=2));print('DONE',len(prov),flush=True)
