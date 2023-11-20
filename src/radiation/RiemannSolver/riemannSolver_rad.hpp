// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef RAD_RIEMANNSOLVER_RIEMANNSOLVER_HPP_
#define RAD_RIEMANNSOLVER_RIEMANNSOLVER_HPP_

#include <string>
#include <memory>

#include "radiation.hpp"
#include "input.hpp"


template <typename Phys>
class RiemannSolver_Rad {
 public:
  // Riemann Solver type

  enum Solver {HLL_RAD};

  RiemannSolver_Rad(Input &input, Radiation<Phys>* radiation);

  template <int> void CalcFlux(IdefixArray4D<real> &);

  //Solver GetSolver() {
  //  return(mySolver);
  //}

  //void ShowConfig();

  // Riemann Solvers
  template<const int>
    void HllRad(IdefixArray4D<real> &);
  
  // Get the right slope limiter
  //template<int dir>
  //ExtrapolateToFaces<Phys, dir>* GetExtrapolator();

 private:
  template <typename P, int dir, PLMLimiter L, int O>
  friend class ExtrapolateToFaces;

  IdefixArray4D<real> Vrad;
  IdefixArray4D<real> Flux;
  IdefixArray3D<real> cMax;
  Radiation<Phys>* radiation;
  DataBlock *data;

  Solver mySolver;

  //std::unique_ptr<ShockFlattening<Phys>> shockFlattening;

  // Because each direction is a different template, we can't use
  //std::unique_ptr<ExtrapolateToFaces<Phys,IDIR>> slopeLimIDIR;
  //std::unique_ptr<ExtrapolateToFaces<Phys,JDIR>> slopeLimJDIR;
  //std::unique_ptr<ExtrapolateToFaces<Phys,KDIR>> slopeLimKDIR;

  //bool haveShockFlattening;
};


template <typename Phys>
RiemannSolver_Rad<Phys>::RiemannSolver_Rad(Input &input, Radiation<Phys>* radiation) : Vrad{radiation->Vrad},
                                      Flux{radiation->FluxRiemann},
                                      cMax{radiation->cMax},
                                      radiation{radiation},
                                      data{radiation->data}
                                      {
  // read Solver from input file
  
  std::string solverString = input.Get<std::string>(std::string(Phys::prefix),"solver",0);
     
  if (solverString.compare("hll_rad") == 0) {
    mySolver = HLL_RAD;
  } else {
    std::stringstream msg;
    msg << "Unknown Radiation solver type " << solverString;
    IDEFIX_ERROR(msg);
  }

  // Shock flattening
  //this->haveShockFlattening = input.CheckEntry(std::string(Phys::prefix),"shockFlattening")>=0;
  // Init shock flattening
  //if(haveShockFlattening) {
  //  this->shockFlattening = std::make_unique<ShockFlattening<Phys>>(
  //                            hydro,input.Get<real>(std::string(Phys::prefix),"shockFlattening",0));
 // }

  // init slope limiters
  //slopeLimIDIR = std::make_unique<ExtrapolateToFaces<Phys,IDIR>>(this);
  //#if DIMENSIONS >= 2
  //slopeLimJDIR = std::make_unique<ExtrapolateToFaces<Phys,JDIR>>(this);
  //#endif
  //#if DIMENSIONS == 3
  //slopeLimKDIR = std::make_unique<ExtrapolateToFaces<Phys,KDIR>>(this);
  //#endif
}

#include "calcFlux.hpp"

#endif //RAD_RIEMANNSOLVER_RIEMANNSOLVER_HPP_