// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef RAD_EVOLVESTAGE_HPP_
#define RAD_EVOLVESTAGE_HPP_

#include "radiation.hpp"

template<typename Phys>
template<int dir>
void Radiation<Phys>::LoopDir(const real t, const real dt) {
    // Step 2: compute the intercell flux with our Riemann solver, store the resulting InvDt
    //this->rSolver->template CalcFlux<dir>(this->FluxRiemann);


    // Step 3: compute the resulting evolution of the conserved variables, stored in Uc
    //CalcRightHandSide<dir>(t,dt);
    
    // Recursive: do next dimension
    if constexpr (dir+1 < DIMENSIONS) LoopDir<dir+1>(t, dt);
}



// Evolve one step forward in time of radiation
template<typename Phys>
void Radiation<Phys>::EvolveStage(const real t, const real dt) {
  idfx::pushRegion("Radiation::EvolveStage");

  // Loop on all of the directions
  LoopDir<IDIR>(t,dt);

  // Step 4: add source terms to the conserved variables (curvature, rotation, etc)
  //if(haveSourceTerms) AddSourceTerms(t, dt);

  idfx::popRegion();
}




#endif //RAD_EVOLVESTAGE_HPP_
