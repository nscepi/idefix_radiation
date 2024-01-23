// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef FLUID_RIEMANNSOLVER_RADSOLVERS_LFRRAD_HPP_
#define FLUID_RIEMANNSOLVER_RADSOLVERS_LFRRAD_HPP_

#include "../idefix.hpp"
#include "fluid.hpp"
#include "extrapolateToFaces.hpp"
#include "flux.hpp"
#include "convertConsToPrim.hpp"
#include "speedRad.hpp"

// Compute Riemann fluxes from states using Lax-Friedrichs-Rusanov solver
template <typename Phys>
template<const int DIR>
void RiemannSolver<Phys>::LFRRad(IdefixArray4D<real> &Flux) {
  idfx::pushRegion("RiemannSolver::LFR_Rad");

  constexpr int ioffset = (DIR==IDIR) ? 1 : 0;
  constexpr int joffset = (DIR==JDIR) ? 1 : 0;
  constexpr int koffset = (DIR==KDIR) ? 1 : 0;

  IdefixArray4D<real> Vc = this->Vc;
  IdefixArray3D<real> cMax = this->cMax;

  // Required for high order interpolations
  IdefixArray1D<real> dx = this->data->dx[DIR];


  ExtrapolateToFaces<Phys,DIR> extrapol = *this->GetExtrapolator<DIR>();

  idefix_for("LFR_Rad_Kernel",
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

      //VWave speeds
      real lambdaL[3];
      real lambdaR[3];


      // 1-- Store the primitive variables on the left, right, and averaged states
      extrapol.ExtrapolatePrimVar(i, j, k, vL, vR);

      real FnormL = std::sqrt(EXPAND(vL[FR1]*vL[FR1] , + vL[FR2]*vL[FR2], + vL[FR3]*vL[FR3]));
      real FnormR = std::sqrt(EXPAND(vR[FR1]*vR[FR1] , + vR[FR2]*vR[FR2], + vR[FR3]*vR[FR3]));

      // Limit the fluxes after extrapolation to satisfy Fr<=Er
      if (FnormL>=vL[ER]) {
        vL[FR1] *= (FnormL == ZERO_F ? ZERO_F : vL[ER]/FnormL); 
        vL[FR2] *= (FnormL == ZERO_F ? ZERO_F : vL[ER]/FnormL);
        FnormL = std::sqrt(EXPAND(vL[FR1]*vL[FR1] , + vL[FR2]*vL[FR2], + vL[FR3]*vL[FR3]));
      }

      if (FnormR>=vR[ER]) {
        vR[FR1] *= (FnormR == ZERO_F ? ZERO_F : vR[ER]/FnormR);
        vR[FR2] *= (FnormR == ZERO_F ? ZERO_F : vR[ER]/FnormR);
        FnormR = std::sqrt(EXPAND(vR[FR1]*vR[FR1] , + vR[FR2]*vR[FR2], + vR[FR3]*vR[FR3]));
      }

      // 2-- Get the wave speed
      real f_paramL = FnormL/vL[ER];
      real f2_paramL = f_paramL*f_paramL;
      real f_paramR = FnormR/vR[ER];
      real f2_paramR = f_paramR*f_paramR;
      
      real cos_thetaL = (FnormL == ZERO_F ? ZERO_F : vL[Xn] / FnormL);
      real cos_thetaR = (FnormR == ZERO_F ? ZERO_F : vR[Xn] / FnormR);

      K_speeds_Rad(lambdaL,f_paramL, f2_paramL, cos_thetaL);
      K_speeds_Rad(lambdaR,f_paramR, f2_paramR, cos_thetaR);

      real lambda_max_L = FMAX(FMAX(lambdaL[0],lambdaL[1]),lambdaL[2]);
      real lambda_max_R = FMAX(FMAX(lambdaR[0],lambdaR[1]),lambdaR[2]);
      real lambda_min_L = FMIN(FMIN(lambdaL[0],lambdaL[1]),lambdaL[2]);
      real lambda_min_R = FMIN(FMIN(lambdaR[0],lambdaR[1]),lambdaR[2]);
      
      real SR = FMAX(lambda_max_L,lambda_max_R);
      real SL = FMIN(lambda_min_L,lambda_min_R);
      
      real cmax  = FMAX(FABS(SL), FABS(SR));

      // 3-- Compute the conservative variables: do this by extrapolation
      K_PrimToCons<Phys>(uL, vL, NULL); 
      K_PrimToCons<Phys>(uR, vR, NULL);

      // 4-- Compute the left and right fluxes (wave speed is null)
      K_Flux<Phys,DIR>(fluxL, vL, uL, 0);
      K_Flux<Phys,DIR>(fluxR, vR, uR, 0);

      // 5-- Compute the flux from the left and right states
#pragma unroll
      for(int nv = 0 ; nv < Phys::nvar; nv++) {
        Flux(nv,k,j,i) = 0.5*(fluxL[nv] + fluxR[nv]-cmax*(uR[nv]-uL[nv]));
      }

      //6-- Compute maximum wave speed for this sweep
      cMax(k,j,i) = cmax;
    }
  );

  idfx::popRegion();
}

#endif // FLUID_RIEMANNSOLVER_RADSOLVERS_LFRRAD_HPP_
