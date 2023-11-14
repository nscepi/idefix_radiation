// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef RAD_CONVERTCONSTOPRIM_HPP_
#define RAD_CONVERTCONSTOPRIM_HPP_

#include "radiation.hpp"


template <typename Phys>
KOKKOS_INLINE_FUNCTION void K_ConsToPrim(real Vrad[], real Urad[]) {
  Vrad[ER] = Urad[ER];

  EXPAND( Vrad[FR1] = Urad[FR1];  ,
          Vrad[FR2] = Urad[FR2];  ,
          Vrad[FR3] = Urad[FR3];  )

}



template <typename Phys>
KOKKOS_INLINE_FUNCTION void K_PrimToCons(real Urad[], real Vrad[]) {
  Urad[ER] = Vrad[ER];

  EXPAND( Urad[FR1] = Vrad[FR1];  ,
          Urad[FR2] = Vrad[FR2];  ,
          Urad[FR3] = Vrad[FR3];  )

}


// Convect Conservative to Primitive variable
template<typename Phys>
void Radiation<Phys>::ConvertConsToPrim() {
  idfx::pushRegion("Radiation::ConvertConsToPrim");

  IdefixArray4D<real> Vrad = this->Vrad;
  IdefixArray4D<real> Urad = this->Urad;

  idefix_for("ConsToPrim",
             0,this->data->np_tot[KDIR],
             0,this->data->np_tot[JDIR],
             0,this->data->np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      real U[Phys::nvar];
      real V[Phys::nvar];

#pragma unroll
      for(int nv = 0 ; nv < Phys::nvar; nv++) {
        U[nv] = Urad(nv,k,j,i);
      }

      K_ConsToPrim<Phys>(V,U);

#pragma unroll
      for(int nv = 0 ; nv<Phys::nvar; nv++) {
        Vrad(nv,k,j,i) = V[nv];
      }
  });

  idfx::popRegion();
}


// Convert Primitive to conservative variables
template<typename Phys>
void Radiation<Phys>::ConvertPrimToCons() {
  idfx::pushRegion("Radiation::ConvertPrimToCons");

  IdefixArray4D<real> Vrad = this->Vrad;
  IdefixArray4D<real> Urad = this->Urad;
  
  idefix_for("ConvertPrimToCons",
             0,this->data->np_tot[KDIR],
             0,this->data->np_tot[JDIR],
             0,this->data->np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      real U[Phys::nvar];
      real V[Phys::nvar];

#pragma unroll
      for(int nv = 0 ; nv < Phys::nvar; nv++) {
        V[nv] = Vrad(nv,k,j,i);
      }

      K_PrimToCons<Phys>(U,V);

#pragma unroll
      for(int nv = 0 ; nv<Phys::nvar; nv++) {
        Urad(nv,k,j,i) = U[nv];
      }
  });

  idfx::popRegion();
}


#endif //RAD_CONVERTCONSTOPRIM_HPP_