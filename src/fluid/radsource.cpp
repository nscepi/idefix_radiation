// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************
#include "radsource.hpp"
#include "physics.hpp"

void RadSource::AddRadSource(const real dt) {
  idfx::pushRegion("RadSource::AddRadSource");

  auto UcGas = this->UcGas;
  auto VcGas = this->VcGas;
  auto UcRad = this->UcRad;
  auto VcRad = this->VcRad;
  auto InvDt = this->InvDt;

  const Type type = this->type;
  real kappa_rad = this->kappa_rad;
  real xi_rad = this->kappa_rad;
  real reduced_c = this->reduced_c;

  idefix_for("RadForce",0,data->np_tot[KDIR],0,data->np_tot[JDIR],0,data->np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      real T4 = std::pow(VcGas(PRS,k,j,i)/VcGas(RHO,k,j,i),4);
      real kk = reduced_c * dt * kappa_rad * VcGas(RHO,k,j,i) ;
      UcRad(ENG,k,j,i) += (VcRad(RHO,k,j,i)+kk*T4)/(ONE_F+kk);

      real xx = reduced_c * dt * xi_rad * VcGas(RHO,k,j,i) ;
      for(int n = MX1 ; n < MX1+COMPONENTS ; n++) {
        UcRad(n,k,j,i) -= VcRad(n,k,j,i)/(ONE_F+xx);
      }
    });
  idfx::popRegion();
}

void RadSource::ShowConfig() {
  idfx::cout << "RadSource: Using ";
  switch(type) {
    case Type::Tconst:
      idfx::cout << "constant Temperature in source term integration";
      break;
    case Type::Tvar:
      idfx::cout << "Temperature solved with additional equation";
      break;
  }
}
