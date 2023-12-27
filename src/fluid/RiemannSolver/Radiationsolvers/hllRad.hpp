// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef FLUID_RIEMANNSOLVER_RADSOLVERS_HLLRAD_HPP_
#define FLUID_RIEMANNSOLVER_RADSOLVERS_HLLRAD_HPP_

#include "../idefix.hpp"
#include "fluid.hpp"
#include "extrapolateToFaces.hpp"
#include "flux.hpp"
#include "convertConsToPrim.hpp"

// Compute Riemann fluxes from states using HLL solver
template <typename Phys>
template<const int DIR>
void RiemannSolver<Phys>::HllRad(IdefixArray4D<real> &Flux) {
  idfx::pushRegion("RiemannSolver::HLL_Rad");

  constexpr int ioffset = (DIR==IDIR) ? 1 : 0;
  constexpr int joffset = (DIR==JDIR) ? 1 : 0;
  constexpr int koffset = (DIR==KDIR) ? 1 : 0;

  IdefixArray4D<real> Vc = this->Vc;
  IdefixArray3D<real> cMax = this->cMax;

  // Required for high order interpolations
  IdefixArray1D<real> dx = this->data->dx[DIR];


  ExtrapolateToFaces<Phys,DIR> extrapol = *this->GetExtrapolator<DIR>();

  idefix_for("HLL_Rad_Kernel",
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

      real zetaL_arg1 = 4.-3.*f2_paramL;
      real zetaL_arg2 = std::sqrt(zetaL_arg1);
      real zetaL_arg3 = 2.*(zetaL_arg1-zetaL_arg2)/3.;
      real zetaL_arg4 = 2.*cos_thetaL*cos_thetaL*(2.-f2_paramL-zetaL_arg2);
      real zetaL = std::sqrt(zetaL_arg3 + zetaL_arg4);

      real zetaR_arg1 = 4.-3.*f2_paramR;
      real zetaR_arg2 = std::sqrt(zetaR_arg1);
      real zetaR_arg3 = 2.*(zetaR_arg1-zetaR_arg2)/3.;
      real zetaR_arg4 = 2.*cos_thetaR*cos_thetaR*(2.-f2_paramR-zetaR_arg2);
      real zetaR = std::sqrt(zetaR_arg3 + zetaR_arg4);

      real epsilonL = 3.+4.*f2_paramL;
      epsilonL /= 5.+2.*zetaL_arg2;
      real epsilonR = 3.+4.*f2_paramR;
      epsilonR /= 5.+2.*zetaR_arg2;

      real lambdaL_1 = f_paramL*cos_thetaL-zetaL;
      lambdaL_1 /= zetaL_arg2;
      real lambdaL_2 = (3*epsilonL-1.)*cos_thetaL;
      lambdaL_2 /= 2.*f_paramL;
      real lambdaL_3 = f_paramL*cos_thetaL+zetaL;
      lambdaL_3 /= zetaL_arg2;

      real lambdaR_1 = f_paramR*cos_thetaR-zetaR;
      lambdaR_1 /= zetaR_arg2;
      real lambdaR_2 = (3*epsilonR-1.)*cos_thetaR;
      lambdaR_2 /= 2.*f_paramR;
      real lambdaR_3 = f_paramR*cos_thetaR+zetaR;
      lambdaR_3 /= zetaR_arg2;
      
      real lambda_max_L = FMAX(FMAX(lambdaL_1,lambdaL_2),lambdaL_3);
      real lambda_max_R = FMAX(FMAX(lambdaR_1,lambdaR_2),lambdaR_3);
      real lambda_min_L = FMIN(FMIN(lambdaL_1,lambdaL_2),lambdaL_3);
      real lambda_min_R = FMIN(FMIN(lambdaR_1,lambdaR_2),lambdaR_3);
      
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
        real dS = SR-SL;
        if(std::abs(dS) < SMALL_NUMBER) {
          dS = SMALL_NUMBER;
//         printf("Velocities are the same\n");
        }
#pragma unroll
        for(int nv = 0 ; nv < Phys::nvar; nv++) {
          Flux(nv,k,j,i) = SL*SR*uR[nv] - SL*SR*uL[nv] + SR*fluxL[nv] - SL*fluxR[nv];
          Flux(nv,k,j,i) /= dS;
        }
      }

      //6-- Compute maximum wave speed for this sweep
      cMax(k,j,i) = cmax;
    }
  );

  idfx::popRegion();
}

#endif // FLUID_RIEMANNSOLVER_RADSOLVERS_HLLRAD_HPP_
