#ifdef FIX_CLASS
FixStyle(vdf/kinetic,FixVdfKinetic)
#else
#ifndef SPARTA_FIX_VDF_KINETIC_H
#define SPARTA_FIX_VDF_KINETIC_H
#include "fix.h"
namespace SPARTA_NS {
class FixVdfKinetic: public Fix {
 public:
  FixVdfKinetic(class SPARTA *,int,char **);
  ~FixVdfKinetic();
  int setmask();
  void init();
  void end_of_step();
 private:
  struct State;
  State *state;
};
}
#endif
#endif
