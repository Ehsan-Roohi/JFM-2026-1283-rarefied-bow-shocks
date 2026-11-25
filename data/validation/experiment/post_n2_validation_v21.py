"""Run after all three completed, archived validation trajectories; never fit a width."""
from pathlib import Path
import hashlib,json,subprocess,sys,tarfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path('campaign')
C=W/'nitrogen_validation_v21_20261031';O=C/'analysis'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins=json.loads((C/'ANALYSIS_PINS.json').read_text())
for p,h in pins.items():assert sha(p)==h,p
camp=json.loads((C/'CAMPAIGN.json').read_text())
for c in camp['configs']:
 receipt=json.loads((Path(c['root'])/'ARCHIVE_VERIFIED.json').read_text());assert receipt['verified']
 assert Path(receipt['archive']).is_file()
# Validate the complete production windows and all provenance before plotting.
subprocess.run(['python3',str(C/'analyze_planar_v22.py'),str(C),str(O)],check=True)
report=json.loads((O/'REPORT.json').read_text())
r=np.genfromtxt(C/'Alsmeyer_N2_M153_digitized_v21.csv',delimiter=',',names=True)
rx=r['x_over_lambda'];ry=r['rho_star'];refmid=float(np.interp(.5,ry,rx))
fig,axes=plt.subplots(1,2,figsize=(11,4.5));refx=rx-refmid
axes[0].errorbar(refx,ry,xerr=r['reading_uncertainty_x'],yerr=r['reading_uncertainty_rho_star'],fmt='s',ms=4,mfc='none',c='black',elinewidth=.5,capsize=1,label='Alsmeyer data, reproduced/digitized')
comparisons={};colors=['#2865a2','#c64f22','#18865f']
for c,color in zip(camp['configs'],colors):
 name=c['name'];b=np.load(O/(name+'.npz'));case=report['cases'][name]
 rh=c['rankine_hugoniot'];mass=camp['models']['baseline']['mass'];rho1=c['nrho']*mass
 x=(b['x']-case['center_m'])/c['transport']['lambda_viscosity_m']
 y=(b['mean'][:,0]-rho1)/(rho1*(rh['compression']-1))
 assert x.min()<refx.min() and x.max()>refx.max()
 predicted=np.interp(refx,x,y);error=predicted-ry
 comparisons[name]={'MAE_rho_star':float(np.mean(abs(error))),'RMS_rho_star':float(np.sqrt(np.mean(error**2))),
  'max_abs_rho_star':float(np.max(abs(error))),'density_width_over_lambda':case['scales']['rho'][1]['gradient_width_m']/c['transport']['lambda_viscosity_m'],
  'shift_only':'Each curve shifted to its own half-density location; neither amplitude nor width fitted.'}
 axes[0].plot(x,y,c=color,label=name.rsplit('_',1)[-1]);axes[1].plot(refx,error,'o-',c=color,ms=3,label=name.rsplit('_',1)[-1])
axes[0].set(xlim=(-8,8),xlabel=r'$(x-x_{\rho,1/2})/\lambda_{\mu,1}$',ylabel=r'$(\rho-\rho_1)/(\rho_2-\rho_1)$')
axes[1].axhline(0,c='gray',lw=.7);axes[1].set(xlabel=r'$(x-x_{\rho,1/2})/\lambda_{\mu,1}$',ylabel='Simulation minus digitized density')
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('N2 M=1.53, upstream300K | unchanged VHS + constant Zrot5\nMarker reading envelope only; no width or collision-parameter fit')
fig.tight_layout()
for ext in ['pdf','png']:fig.savefig(O/f'Supplement_N2_planar_experimental_validation.{ext}',dpi=170,bbox_inches='tight')
plt.close(fig)
result={'comparisons':comparisons,'reference_provenance':json.loads((C/'REFERENCE_PROVENANCE_v21.json').read_text()),
 'reference_origin_shift_lambda':refmid,'experimental_validation_pass':None,'scientific_convergence':False,
 'review_required':'Inspect discrepancy alongside sampling, mesh/time/domain sensitivity and the reproduced-marker reading envelope. No acceptance threshold has been invented.',
 'scope':'Independent low-enthalpy density-profile comparison of the fixed baseline nitrogen model, not validation of all cylinder temperatures, chemistry, surface interaction or high-Mach physics.'}
(O/'VALIDATION_REPORT.json').write_text(json.dumps(result,indent=2))
for p in [C/'analyze_planar_v22.py',Path(__file__),C/'ANALYSIS_PINS.json',C/'Alsmeyer_N2_M153_digitized_v21.csv',C/'REFERENCE_PROVENANCE_v21.json']:
 import shutil;shutil.copy2(p,O/p.name)
dest=Path(camp['archive']);dest.mkdir(parents=True,exist_ok=True);archive=dest/'validation_analysis.tar';assert not archive.exists()
files=[p for p in O.iterdir() if p.is_file()];members={p.name:sha(p) for p in files}
with tarfile.open(archive,'w') as t:
 for p in files:t.add(p,arcname=p.name,recursive=False)
with tarfile.open(archive) as t:
 for n,h in members.items():assert hashlib.file_digest(t.extractfile(n),'sha256').hexdigest()==h
receipt={'archive':str(archive),'sha256':sha(archive),'verified_members':members,'verified':True}
(O/'ARCHIVE_VERIFIED.json').write_text(json.dumps(receipt,indent=2));(dest/'analysis_ARCHIVE_VERIFIED.json').write_text(json.dumps(receipt,indent=2))
print('VALIDATION_ANALYSIS_COMPLETE',O,flush=True)
