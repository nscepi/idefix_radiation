// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef RAD_RIEMANNSOLVER_CALCFLUX_HPP_
#define RAD_RIEMANNSOLVER_CALCFLUX_HPP_

#include "hllRad.hpp"

//#include "shockFlattening.hpp"

// Compute Riemann fluxes from states
template <typename Phys>
template <int dir>
void RiemannSolver_Rad<Phys>::CalcFlux(IdefixArray4D<real> &flux) {
  idfx::pushRegion("RiemannSolver_Rad::CalcFlux");
  //if constexpr(dir == IDIR) {
    // enable shock flattening
    //if(haveShockFlattening) shockFlattening->FindShock();
  //}

  
switch (mySolver) {
    case HLL:
      HllHD<dir>(flux);
      break;
    default: // do nothing
        IDEFIX_ERROR("Internal error: Unknown solver");
        break;
}

  idfx::popRegion();
}
#endif // RAD_RIEMANNSOLVER_CALCFLUX_HPP_
