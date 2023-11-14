// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef RAD_ENROLL_HPP_
#define RAD_ENROLL_HPP_

#include "dataBlock.hpp"

template<typename Phys>
void Radiation<Phys>::ResetStage() {
  // Reset variables required at the beginning of each stage
  // (essentially linked to timestep evaluation)
  idfx::pushRegion("Radiation::ResetStage");

  IdefixArray3D<real> InvDt=this->InvDt;

  idefix_for("RadiationResetStage",0,this->data->np_tot[KDIR],0,this->data->np_tot[JDIR],0,this->data->np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      InvDt(k,j,i) = ZERO_F;
  });

  idfx::popRegion();
}

#endif //RAD_ENROLL_HPP_