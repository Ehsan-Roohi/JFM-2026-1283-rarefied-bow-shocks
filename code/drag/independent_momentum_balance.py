"""An independent streamwise drag balance from line normals and molecular momentum transfer."""
import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys

def one(dump):
    # For the oriented upper boundary in the native mesh, (dy,-dx)/length
    # points into the solid. The traction on the solid is pressure times this normal.
    force=[];normal=[];shear=[];native_pressure=[];native_shear=[]
    with dump.open(encoding='ascii') as f:
        while True:
            header=f.readline()
            if not header:raise ValueError(dump)
            if header.startswith('ITEM: SURFS '):break
        assert header.split()[2:8]==['id','v1x','v1y','v2x','v2y','area']
        for line in f:
            _,x1,y1,x2,y2,ds,n,fx,fy,px,py,sx,sy,p=map(float,line.split())
            dx=x2-x1;dy=y2-y1
            assert math.isclose(ds,math.hypot(dx,dy),rel_tol=1e-11,abs_tol=1e-14)
            force.append(fx);normal.append(p*dy)
            shear.append((fx*dx+fy*dy)*dx/(ds*ds))
            native_pressure.append(px*ds);native_shear.append(sx*ds)
    F,Fn,Ft,Fnp,Fsp=(math.fsum(a) for a in (force,normal,shear,native_pressure,native_shear))
    assert F>0
    residual=max(abs(F-Fn-Ft),abs(Fn-Fnp),abs(Ft-Fsp))
    assert residual<1e-10*F,(dump,F,Fn,Ft,Fnp,Fsp)
    return F,Fn,Ft,residual/F

def summarize(root, kn):
    record=json.loads((root/('closed-'+kn+'.json')).read_text())
    rows={}
    for name,selected in record['selected_blocks'].items():
        ev=root/'evidence'
        settings=ev/'parent/cases'/name
        native=(settings/'in.case').read_text()
        line=next(l.split() for l in native.splitlines() if l.startswith('mixture gas '))
        g=line[2];dens=float(line[line.index('nrho')+1]);u=float(line[line.index('vstream')+1])
        sp=next(l.split()[1] for l in native.splitlines() if l.startswith('species '))
        mass=float(next(l.split()[2] for l in (settings/sp).resolve().read_text().splitlines() if l.startswith(g+' ')))
        q=.5*dens*mass*u*u
        D=.3048
        nb=[];pn=[];sn=[];errs=[]
        for k in range(1,11):
            block=selected[str(k)]
            a=ev/'cases'/name/'attempts'/block['attempt']
            sums=[one(a/f) for f in block['dumps']]
            assert len(sums)==5
            for i,target in enumerate((nb,pn,sn)):
                target.append(2*statistics.mean(v[i] for v in sums)/(q*D))
            errs.extend(v[3] for v in sums)
        rows[name]={'case':name,'q_Pa':q,'mean_Cd':statistics.mean(nb),'normal_Cd':statistics.mean(pn),'shear_Cd':statistics.mean(sn),'drag_N_per_m':statistics.mean(nb)*q*D,'block_sd':statistics.stdev(nb),'first5_Cd':statistics.mean(nb[:5]),'last5_Cd':statistics.mean(nb[5:]),'max_relative_projection_closure':max(errs)}
    return rows

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    rows={}
    for kn in sys.argv[1:] or ['Kn0p01','Kn0p1']:
        rows.update(summarize(root,kn))
    print(json.dumps(rows,indent=2))
