#include "fix_vdf_kinetic.h"
#include "vdf_kernel.h"
#include "domain.h"
#include "error.h"
#include "grid.h"
#include "input.h"
#include "particle.h"
#include "update.h"
#include <mpi.h>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <memory>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

using namespace SPARTA_NS;
namespace jf=jfmvdf;
namespace {
std::string slurp(const std::string &path) {
  std::ifstream f(path.c_str(),std::ios::binary);
  jf::require(f.good(),"cannot open VDF input or existing record");
  std::ostringstream s;s<<f.rdbuf();jf::require(!f.bad(),"cannot read VDF input or existing record");return s.str();
}
void immutable(const std::string &name,const std::string &bytes) {
  std::string pattern=name+".part.XXXXXX";std::vector<char> temp(pattern.begin(),pattern.end());temp.push_back(0);
  int fd=mkstemp(temp.data());jf::require(fd>=0,"cannot create VDF temporary output in the pre-existing segment directory");
  try {
    size_t off=0;while(off<bytes.size()){
      ssize_t n=write(fd,bytes.data()+off,bytes.size()-off);
      if(n<0 && errno==EINTR) continue;
      jf::require(n>0,"VDF output write failed");off+=static_cast<size_t>(n);
    }
    jf::require(fsync(fd)==0,"VDF data fsync failed");
    int rc=close(fd);fd=-1;jf::require(rc==0,"VDF output close failed");
    if(link(temp.data(),name.c_str())!=0) {
      jf::require(errno==EEXIST,"VDF atomic publication failed");
      jf::require(slurp(name)==bytes,"different VDF sample or configuration already exists at this location");
    }
    jf::require(unlink(temp.data())==0,"VDF temporary output removal failed");
    const std::string dir=name.substr(0,name.find_last_of('/'));
    int d=open(dir.c_str(),O_RDONLY|O_DIRECTORY);
    if(d>=0){int r=fsync(d);int e=errno;close(d);jf::require(r==0||e==EINVAL||e==EOPNOTSUPP,"VDF directory fsync failed");}
  } catch(...) {if(fd>=0) close(fd);unlink(temp.data());throw;}
}
template<class T> void emit(std::string &out,const T &value) {
  out.append(reinterpret_cast<const char*>(&value),sizeof(value));
}
std::string basename_step(uint64_t step) {
  std::ostringstream s;s<<"sample_"<<std::setw(15)<<std::setfill('0')<<step<<".vdf";return s.str();
}
std::vector<jf::Molecule> gather_molecules(jf::Heap &local,unsigned cap,MPI_Comm comm,int rank,int ranks) {
  static_assert(sizeof(jf::Molecule)==96,"unexpected raw molecule layout");
  std::vector<jf::Molecule> items=jf::take_sorted(local);
  if(rank) {
    int n=static_cast<int>(items.size());MPI_Send(&n,1,MPI_INT,0,11,comm);
    if(n) MPI_Send(items.data(),n*sizeof(jf::Molecule),MPI_BYTE,0,12,comm);
    return std::vector<jf::Molecule>();
  }
  for(const auto &m:items) jf::retain(local,cap,m);
  for(int peer=1;peer<ranks;++peer) {
    int n=0;MPI_Recv(&n,1,MPI_INT,peer,11,comm,MPI_STATUS_IGNORE);
    jf::require(n>=0 && unsigned(n)<=cap,"invalid molecular sample message");
    items.resize(n);
    if(n) MPI_Recv(items.data(),n*sizeof(jf::Molecule),MPI_BYTE,peer,12,comm,MPI_STATUS_IGNORE);
    for(const auto &m:items) jf::retain(local,cap,m);
  }
  return jf::take_sorted(local);
}
}
struct FixVdfKinetic::State {
  MPI_Comm comm;
  int rank,ranks,ispecies;
  bigint first,every;
  std::string dir,manifest;
  jf::Config cfg;
  jf::Layout layout;
  jf::Tally local,global;
  std::vector<unsigned> mask;
  bool ready;
  State(const std::string &text,const std::string &out,bigint start,bigint interval,MPI_Comm world):
     comm(MPI_COMM_NULL),rank(0),ranks(0),ispecies(-1),first(start),every(interval),dir(out),manifest(text),cfg(jf::parse(text)),layout(cfg),
     local(cfg.stations.size(),layout),global(cfg.stations.size(),layout),ready(false) {
    MPI_Comm_dup(world,&comm);MPI_Comm_rank(comm,&rank);MPI_Comm_size(comm,&ranks);
  }
  ~State(){if(comm!=MPI_COMM_NULL) MPI_Comm_free(&comm);}
};
FixVdfKinetic::FixVdfKinetic(SPARTA *sparta,int narg,char **arg):Fix(sparta,narg,arg),state(NULL) {
  if(narg!=6) error->all(FLERR,"Usage: fix ID vdf/kinetic MANIFEST EXISTING_SEGMENT_DIRECTORY FIRST_STEP EVERY_NSTEPS");
  bigint first=input->bnumeric(FLERR,arg[4]), every=input->bnumeric(FLERR,arg[5]);
  if(first<0 || every<=0) error->all(FLERR,"Invalid VDF sampling schedule");
  // SPARTA applies nevery modulo global timestep before calling end_of_step.
  nevery=1;
  try{
    int rank,n=0;MPI_Comm_rank(world,&rank);std::string text;
    if(rank==0){text=slurp(arg[2]);jf::require(!text.empty() && text.size()<131072,"empty or oversized VDF manifest");n=int(text.size());}
    MPI_Bcast(&n,1,MPI_INT,0,world);text.resize(n);MPI_Bcast(&text[0],n,MPI_CHAR,0,world);
    state=new State(text,arg[3],first,every,world);
  }catch(const std::exception &e){error->one(FLERR,e.what());}
}
FixVdfKinetic::~FixVdfKinetic(){delete state;}
int FixVdfKinetic::setmask(){return END_OF_STEP;}
void FixVdfKinetic::init() {
  State &q=*state;
  try{
    jf::require(domain->dimension==q.cfg.dimension && !domain->axisymmetric,"VDF manifest and native Cartesian dimension disagree");
    jf::require(update->unit_style && std::string(update->unit_style)=="si","VDF energies require SI solver units");
    q.ispecies=particle->find_species(const_cast<char*>(q.cfg.species.c_str()));
    jf::require(q.ispecies>=0 && particle->nspecies==1,"VDF case must have exactly the named species");
    jf::require(particle->species[q.ispecies].mass>0 && update->fnum>0,"invalid species mass or fnum");
    jf::require(!q.cfg.rotation || particle->species[q.ispecies].rotdof>0,"rotational statistics requested for inactive species DOF");
    if(!q.ready && q.rank==0) {
      immutable(q.dir+"/stations.manifest",q.manifest);
      std::ostringstream desc;desc<<std::setprecision(17)
        <<"JFM_VDF_RUN_2\nfirst "<<q.first<<"\nevery "<<q.every<<"\ndt "<<update->dt
        <<"\nfnum "<<update->fnum<<"\nmass "<<particle->species[q.ispecies].mass
        <<"\ncellweight "<<grid->cellweightflag<<"\ncellint_bytes "<<sizeof(cellint)<<"\n";
      immutable(q.dir+"/run.meta",desc.str());
    }
    q.ready=true;
  }catch(const std::exception &e){error->one(FLERR,e.what());}
}
void FixVdfKinetic::end_of_step() {
  State &q=*state;const bigint t=update->ntimestep;
  if(t<q.first || (t-q.first)%q.every) return;
  try{
    const auto &c=q.cfg;const auto &l=q.layout;const int ns=int(c.stations.size());
    const bool snapshot=c.cap>0 && (((t-q.first)/q.every+1)%c.samples_per_block==0);
    q.local.clear(); std::vector<jf::Heap> samples(ns);
    auto observe=[&](int ip,int ic,unsigned bits) {
      const auto &p=particle->particles[ip];
      if(p.ispecies!=q.ispecies) return;
      bool matched=false;
      for(int j=0;j<ns;++j) if((bits&(1u<<j)) && c.stations[j].contains(p.x,c.dimension)){
        jf::require(!matched,"overlapping VDF stations would double-count an observation");matched=true;
        const double pw=update->fnum*(grid->cellweightflag?grid->cinfo[ic].weight:1.);
        const double er=c.rotation?p.erot:0.;
        jf::accumulate(c,l,q.local,j,p.v,pw,particle->species[q.ispecies].mass,er);
        if(snapshot){
          const uint64_t key=jf::sample_key(c.salt,uint64_t(t),j,q.rank,ip);
          if(samples[j].size()<unsigned(c.cap) || key<samples[j].top().key){
            jf::Molecule m{};m.key=key;m.cell=uint64_t(grid->cells[ic].id);m.rank=q.rank;m.index=ip;m.particle_id=p.id;
            std::copy(p.x,p.x+3,m.x);std::copy(p.v,p.v+3,m.v);m.erot=er;m.weight=pw;
            jf::retain(samples[j],c.cap,m);
          }
        }
      }
    };
    // Geometry is refreshed after migration/load balancing without changing solver state.
    // SPARTA keeps valid particle lists after ordinary collisions. If invalid, use only
    // native rank-owned particles and an inexpensive cell mask.
    const bool listed=particle->sorted && particle->next;
    if(!listed) q.mask.assign(grid->nlocal,0);
    bool eligible_rank=false;
    for(int ic=0;ic<grid->nlocal;++ic) {
      const auto &cell=grid->cells[ic];const auto &info=grid->cinfo[ic];
      if(info.volume<=0 || cell.nsplit>1) continue;
      unsigned bits=0;
      for(int j=0;j<ns;++j) if(c.stations[j].overlaps(cell.lo,cell.hi,c.dimension)) bits|=1u<<j;
      if(!bits) continue;
      eligible_rank=true;
      if(!listed){q.mask[ic]=bits;continue;}
      int ip=info.first;
      for(int n=0;n<info.count;++n){
        jf::require(ip>=0 && ip<particle->nlocal && particle->particles[ip].icell==ic,"invalid native particle list in a VDF station cell");
        observe(ip,ic,bits);ip=particle->next[ip];
      }
      jf::require(ip==-1,"inconsistent native station cell list length");
    }
    if(eligible_rank && !listed) for(int ip=0;ip<particle->nlocal;++ip){
      int ic=particle->particles[ip].icell;
      jf::require(ic>=0 && ic<grid->nlocal,"VDF particle has no owned cell");
      if(q.mask[ic]) observe(ip,ic,q.mask[ic]);
    }
    int local_activity[2]={eligible_rank?1:0,eligible_rank && listed?1:0},activity[2]={0,0};
    MPI_Reduce(local_activity,activity,2,MPI_INT,MPI_SUM,0,q.comm);
    MPI_Reduce(q.local.raw.data(),q.global.raw.data(),int(q.local.raw.size()),MPI_UNSIGNED_LONG_LONG,MPI_SUM,0,q.comm);
    MPI_Reduce(q.local.weight.data(),q.global.weight.data(),int(q.local.weight.size()),MPI_DOUBLE,MPI_SUM,0,q.comm);
    MPI_Reduce(q.local.moments.data(),q.global.moments.data(),int(q.local.moments.size()),MPI_DOUBLE,MPI_SUM,0,q.comm);
    std::string out;
    if(q.rank==0) {
      uint32_t endian=1;jf::require(*reinterpret_cast<unsigned char*>(&endian)==1 && sizeof(double)==8,"VDF2 requires IEEE binary64 on a little-endian host");
      out="JFMVDF02"; emit(out,uint64_t(t));
      emit(out,double(update->time+(t-update->time_last_update)*update->dt));
      emit(out,uint32_t(ns));emit(out,uint32_t(l.nhist));emit(out,uint32_t(l.ne));
      emit(out,uint32_t(snapshot));emit(out,uint32_t(q.ranks));emit(out,uint32_t(activity[0]));emit(out,uint32_t(activity[1]));
    }
    for(int j=0;j<ns;++j) {
      std::vector<jf::Molecule> mol;
      if(snapshot) mol=gather_molecules(samples[j],c.cap,q.comm,q.rank,q.ranks);
      if(q.rank) continue;
      int start=j*l.nhist; uint32_t used=0;
      for(int b=0;b<l.nhist;++b) if(q.global.raw[start+b]) ++used;
      emit(out,used);
      for(int b=0;b<l.nhist;++b) if(q.global.raw[start+b]) {
        emit(out,uint32_t(b));emit(out,uint64_t(q.global.raw[start+b]));emit(out,q.global.weight[start+b]);
      }
      for(int b=0;b<l.ne;++b) emit(out,q.global.moments[j*l.ne+b]);
      if(snapshot) jf::require(mol.size()==std::min<uint64_t>(c.cap,q.global.raw[start]),"incomplete distributed molecular sample");
      emit(out,uint32_t(mol.size()));
      for(const auto &m:mol) {
        emit(out,m.key);emit(out,m.cell);emit(out,m.rank);emit(out,m.index);emit(out,m.particle_id);
        for(double v:m.x) emit(out,v);for(double v:m.v) emit(out,v);emit(out,m.erot);emit(out,m.weight);
      }
    }
    if(q.rank==0) immutable(q.dir+"/"+basename_step(uint64_t(t)),out);
  }catch(const std::exception &e){error->one(FLERR,e.what());}
}
