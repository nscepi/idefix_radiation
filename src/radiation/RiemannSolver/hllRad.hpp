// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef RAD_RIEMANNSOLVER_HLLRAD_HPP_
#define RAD_RIEMANNSOLVER_HLLRAD_HPP_

#include "../idefix.hpp"
#include "radiation.hpp"
#include "convertConsToPrim.hpp"


// Compute Riemann fluxes from states using HLL solver for radiation
template <typename Phys>
template<const int DIR>
void RiemannSolver_Rad<Phys>::HllRad(IdefixArray4D<real> &Flux) {
  idfx::pushRegion("RiemannSolver_Rad::HLLRad_Solver");

  constexpr int ioffset = (DIR==IDIR) ? 1 : 0;
  constexpr int joffset = (DIR==JDIR) ? 1 : 0;
  constexpr int koffset = (DIR==KDIR) ? 1 : 0;

  IdefixArray4D<real> Vrad = this->Vrad;
  IdefixArray3D<real> cMax = this->cMax;

  ExtrapolateToFaces<Phys,DIR> extrapol = *this->GetExtrapolator<DIR>();

  ExtrapolateToFaces<Phys,DIR> extrapol = *this->GetExtrapolator<DIR>();
  idefix_for("HLL_Kernel",
             data->beg[KDIR],data->end[KDIR]+koffset,
             data->beg[JDIR],data->end[JDIR]+joffset,
             data->beg[IDIR],data->end[IDIR]+ioffset,
    KOKKOS_LAMBDA (int k, int j, int i) {
      // Init the directions (should be in the kernel for proper optimisation by the compilers)
      constexpr int Xn = DIR+MX1;

      // Primitive variables
      real vL[Phys::nvar];
      real vR[Phys::nvar];

      // Conservative variables
      real uL[Phys::nvar];
      real uR[Phys::nvar];

      // Flux (left and right)
      real fluxL[Phys::nvar];
      real fluxR[Phys::nvar];

      // Signal speeds
      real cL, cR, cmax;

      // 1-- Store the primitive variables on the left, right, and averaged states
      extrapol.ExtrapolatePrimVar(i, j, k, vL, vR);

      // 2-- Get the wave speed
      cL = HALF_F*(GetWaveSpeed(k,j,i)+GetWaveSpeed(k-koffset,j-joffset,i-ioffset));
      cR = cL;

      // 4.1
      real cminL = vL[Xn] - cL;
      real cmaxL = vL[Xn] + cL;

      real cminR = vR[Xn] - cR;
      real cmaxR = vR[Xn] + cR;

      real SL = FMIN(cminL, cminR);
      real SR = FMAX(cmaxL, cmaxR);

      cmax  = FMAX(FABS(SL), FABS(SR));

      // 2-- Compute the conservative variables: do this by extrapolation
      K_PrimToCons<Phys>(uL, vL, &eos);
      K_PrimToCons<Phys>(uR, vR, &eos);

      // 3-- Compute the left and right fluxes
      K_Flux<Phys,DIR>(fluxL, vL, uL, cL*cL);
      K_Flux<Phys,DIR>(fluxR, vR, uR, cR*cR);

      // 5-- Compute the flux from the left and right states
      if (SL > 0) {
#pragma unroll
        for (int nv = 0 ; nv < Phys::nvar; nv++) {
          Flux(nv,k,j,i) = fluxL[nv];
        }
      } else if (SR < 0) {
#pragma unroll
        for (int nv = 0 ; nv < Phys::nvar; nv++) {
          Flux(nv,k,j,i) = fluxR[nv];
        }
      } else {
#pragma unroll
        for(int nv = 0 ; nv < Phys::nvar; nv++) {
          Flux(nv,k,j,i) = SL*SR*uR[nv] - SL*SR*uL[nv] + SR*fluxL[nv] - SL*fluxR[nv];
          Flux(nv,k,j,i) /= (SR - SL);
        }
      }

      //6-- Compute maximum wave speed for this sweep
      cMax(k,j,i) = cmax;
    }
  );

  idfx::popRegion();
}






#endif //RAD_RIEMANNSOLVER_HLLRAD_HPP_