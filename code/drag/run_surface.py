"""Surface force continuations for the eight matched Mach-10 cases."""
from pathlib import Path
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
BODIES = ('cylinder', 'plate')
GASES = ('Ar', 'N2')
BLOCKS = 10
SUBBLOCKS = 5
MARKER = re.compile(r'^DRAG_BLOCK_COMMITTED (\d+) (\d+) (\d+) (state[01]\.restart)$', re.M)

def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as fp:
        for part in iter(lambda: fp.read(4 * 1024 * 1024), b''):
            value.update(part)
    return value.hexdigest()

def state_stamp(path):
    st=path.stat()
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns]

def write_json(path, value):
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(str(tmp),str(path))

def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def physics(case, conf):
    lines=(case/'in.case').read_text().splitlines()
    select=('mixture gas ', 'collide ', 'collide_modify ', 'surf_collide ', 'surf_modify ', 'fix inflow ', 'timestep ', 'fix lb ')
    actual=[]
    for prefix in select:
        found=[x.strip() for x in lines if x.strip().startswith(prefix)]
        if len(found)!=1:
            raise ValueError((prefix,found))
        line=found[0]
        if prefix=='collide ':
            tokens=line.split()
            tokens[3]=str((case/tokens[3]).resolve())
            line=' '.join(tokens)
        actual.append(line)
    if 'fix inflow emit/face gas xlo xhi yhi twopass' not in actual or 'surf_collide wall diffuse 500.0 1.0' not in actual:
        raise ValueError('Unexpected surface or inflow model')
    step=float(next(x for x in actual if x.startswith('timestep ')).split()[1])
    assert math.isclose(step,conf['dt'],rel_tol=1e-12)
    return actual

def previous(attempts,case_meta):
    end=case_meta['total_steps']; width=case_meta['block_steps']
    candidates=[]
    for folder in sorted(attempts.glob('attempt-*')):
        log=folder/'log.drag'
        admission=folder/'startup.json'
        if not log.exists() or not admission.exists():
            continue
        record=json.loads(admission.read_text())
        for match in MARKER.finditer(log.read_text(errors='replace')):
            k,step,np=map(int,match.groups()[:3])
            if step!=end+(k+1)*width or not (1<=k<=BLOCKS) or np<=0:
                raise ValueError('Invalid committed restart marker')
            candidates.append((k,step,np,folder/match.group(4),record['attempt']))
    if not candidates:
        return None
    return max(candidates,key=lambda x:(x[0],x[4]))

def create_deck(config, physical, source, initial_k, source_step, source_np, seed):
    b=config['block_steps']; sub=b//SUBBLOCKS
    assert b%SUBBLOCKS==0 and math.isclose(b*config['dt'],0.0003,rel_tol=1e-9)
    lines=[
        'seed '+str(seed), 'read_restart '+str(source)+' balance rcb cell',
        *physical, 'stats '+str(sub), 'stats_style step time dt cpu np nscoll', 'run 0',
        'variable dragstep equal step', 'variable dragnp equal np',
        'variable dragtime equal time', 'variable dragdt equal dt',
        'print "DRAG_ADMISSION '+ ' '.join('$'+'{'+name+'}' for name in ('dragstep','dragnp','dragtime','dragdt'))+'"',
        'if "$'+'{dragstep} != '+str(source_step)+'" then "quit 31"',
        'if "$'+'{dragnp} != '+str(source_np)+'" then "quit 32"',
    ]
    if initial_k==0:
        lines+=['run '+str(b)]
    lines+=['compute hits surf all gas n fx fy px py shx shy press',
            'fix wall ave/surf all 1 '+str(sub)+' '+str(sub)+' c_hits[*]',
            'run 1',
            'dump wallDump surf all '+str(sub)+' wall.*.surf id v1x v1y v2x v2y area '+' '.join('f_wall['+str(i)+']' for i in range(1,9)),
            'dump_modify wallDump format float %.16g']
    for k in range(initial_k+1,BLOCKS+1):
        filename='state'+str(k%2)+'.restart'
        lines+=['run '+str(b-(1 if k==initial_k+1 else 0)),
                'write_restart '+filename,
                'print "DRAG_BLOCK_COMMITTED '+str(k)+' $'+'{dragstep} $'+'{dragnp} '+filename+'"']
    lines+=['print "JFM_DRAG_SAMPLING_COMPLETE"']
    return '\n'.join(lines)+'\n'

def work(namespace, case, config, case_out):
    attempts=case_out/'attempts'
    attempts.mkdir(exist_ok=True)
    prior=previous(attempts,config)
    if prior and prior[0]==BLOCKS:
        print('Ten closed force blocks already exist',flush=True)
        return
    parent=case/'checkpoint.final'
    original_log=(case/'log.case').read_text(errors='replace')
    if not re.search(r'^CURVATURE_RUN_COMPLETE '+str(config['total_steps'])+r'$',original_log,re.M):
        raise ValueError('Original case did not close at the registered step')
    rows=re.findall(r'^\s*'+str(config['total_steps'])+r'\s+[\d.eE+-]+\s+(\d+)\s+\d+\s+\d+\s+\d+\s*$',original_log,re.M)
    if not rows:raise ValueError('Missing final particle count from the parent run')
    k,step,np,source=(prior[:4] if prior else (0,config['total_steps'],int(rows[-1]),parent))
    attempt=1+max([int(d.name.split('-')[-1]) for d in attempts.glob('attempt-*')] or [0])
    folder=attempts/('attempt-%04d'%attempt)
    folder.mkdir()
    index=GASES.index(config['gas'])*4+BODIES.index(config['body'])*2+(config['Kn']==0.1)
    seed=31507001+1000*index+attempt
    before=state_stamp(source); h=sha(source)
    if before!=state_stamp(source):raise RuntimeError('Restart changed during the full digest')
    expected=json.loads((ROOT/'pins.json').read_text())
    pinned=expected['cases'][case.name]
    for name in ['in.case','case.json']:
        assert sha(case/name)==pinned[name],name
    for name, digest in expected['shared'].items():
        assert sha(case.parent.parent/name)==digest,name
    assert sha(namespace.binary)==expected['binary_sha256']
    if not prior:assert before[1:]==pinned['restart_stamp'][1:]
    ntasks=int(os.environ['SLURM_NTASKS'])
    assert ntasks==config['ntasks']
    physical=physics(case,config)
    deck=create_deck(config,physical,source,k,step,np,seed)
    (folder/'in.drag').write_text(deck)
    record={'case':case.name,'attempt':attempt,'job_id':os.environ['SLURM_JOB_ID'],
            'restart_count':os.environ.get('SLURM_RESTART_COUNT','0'),'utc':timestamp(),
            'source':str(source),'source_sha256':h,'source_stamp':before,'source_step':step,'source_particles':np,
            'source_closed_blocks':k,'original_parent_job':pinned['original_job'],'seed':seed,'MPI_ranks':ntasks,
            'binary_sha256':expected['binary_sha256'],'deck_sha256':sha(folder/'in.drag'),
            'sample_blocks':BLOCKS,'subblocks_per_block':SUBBLOCKS,'sample_every_solver_step':True}
    write_json(folder/'startup.json',record)
    run=[str(namespace.mpirun),'-np',str(ntasks),str(namespace.binary),'-in','in.drag','-log','log.drag']
    print(case.name,'attempt',attempt,'restart step',step,'N',np,'seed',seed,flush=True)
    with (folder/'screen.txt').open('w') as screen:
        result=subprocess.run(run,cwd=str(folder),stdout=screen,stderr=subprocess.STDOUT)
    text=(folder/'log.drag').read_text(errors='replace') if (folder/'log.drag').exists() else ''
    marks=MARKER.findall(text)
    valid=result.returncode==0 and len(marks)==BLOCKS-k and re.search(r'^JFM_DRAG_SAMPLING_COMPLETE$',text,re.M) is not None
    write_json(folder/'exit.json',{'utc':timestamp(),'returncode':result.returncode,'closure_markers':marks,'complete':valid})
    if not valid:
        print((folder/'screen.txt').read_text(errors='replace')[-1800:],flush=True)
        raise SystemExit(result.returncode or 24)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('case')
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--mpirun',type=Path,required=True)
    args=parser.parse_args()
    case=args.parent/'cases'/args.case
    conf=json.loads((case/'case.json').read_text())
    assert conf['gas'] in GASES and conf['body'] in BODIES and conf['Kn'] in (0.01,0.1)
    dest=ROOT/'cases'/args.case
    with (dest/'writer.lock').open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        work(args,case,conf,dest)
if __name__=='__main__':main()
