// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef FLUID_RIEMANNSOLVER_RADSOLVERS_HLLCRAD_HPP_
#define FLUID_RIEMANNSOLVER_RADSOLVERS_HLLCRAD_HPP_

#include "../idefix.hpp"
#include "fluid.hpp"
#include "extrapolateToFaces.hpp"
#include "flux.hpp"
#include "convertConsToPrim.hpp"
#include "speedRad.hpp"
#include "lim_fluxRad.hpp"

// Compute Riemann fluxes from states using HLLC solver
template <typename Phys>
template<const int DIR>
void RiemannSolver<Phys>::HllcRad(IdefixArray4D<real> &Flux) {
  idfx::pushRegion("RiemannSolver::HLL_Rad");

  constexpr int ioffset = (DIR==IDIR) ? 1 : 0;
  constexpr int joffset = (DIR==JDIR) ? 1 : 0;
  constexpr int koffset = (DIR==KDIR) ? 1 : 0;
  
  IdefixArray4D<real> Vc = this->Vc;
  IdefixArray3D<real> cMax = this->cMax;

  // Required for high order interpolations
  IdefixArray1D<real> dx = this->data->dx[DIR];


  ExtrapolateToFaces<Phys,DIR> extrapol = *this->GetExtrapolator<DIR>();

  idefix_for("HLLC_Rad_Kernel",
             data->beg[KDIR],data->end[KDIR]+koffset,
             data->beg[JDIR],data->end[JDIR]+joffset,
             data->beg[IDIR],data->end[IDIR]+ioffset,
    KOKKOS_LAMBDA (int k, int j, int i) {
      // Init the directions (should be in the kernel for proper optimisation by the compilers)
      EXPAND( constexpr int Xn = DIR+MX1;                    ,
              constexpr int Xt = (DIR == IDIR ? MX2 : MX1);  ,
              constexpr int Xb = (DIR == KDIR ? MX2 : MX3);  )
      
      constexpr int ioffset = (DIR==IDIR ? 1 : 0);
      constexpr int joffset = (DIR==JDIR ? 1 : 0);
      constexpr int koffset = (DIR==KDIR ? 1 : 0);
      
      // Reduced velocity of light
      real reduced_c = 1.;
      
      // Primitive variables
      real vL[Phys::nvar];
      real vR[Phys::nvar];

      // Conservative variables
      real uL[Phys::nvar], usL[Phys::nvar];
      real uR[Phys::nvar], usR[Phys::nvar];

      // Flux (left and right)
      real fluxL[Phys::nvar];
      real fluxR[Phys::nvar];

      //Wave speeds
      real lambdaL[2];
      real lambdaR[2];
      
      // 1-- Store the primitive variables on the left, right, and averaged states
      extrapol.ExtrapolatePrimVar(i, j, k, vL, vR);

      // Limit Fr after extrapolation to satisfy Fr<=Er
      K_limit_RadFlux(vL);
      K_limit_RadFlux(vR);

      // 2-- Get the wave speed
      K_speeds_Rad(lambdaL,vL,Xn);
      K_speeds_Rad(lambdaR,vR,Xn);
 
      real lambda_max_L = FMAX(lambdaL[0],lambdaL[1]);
      real lambda_max_R = FMAX(lambdaR[0],lambdaR[1]);
      real lambda_min_L = FMIN(lambdaL[0],lambdaL[1]);
      real lambda_min_R = FMIN(lambdaR[0],lambdaR[1]);
      
      real SR = FMAX(ZERO_F,FMAX(lambda_max_L,lambda_max_R));
      real SL = FMIN(ZERO_F,FMIN(lambda_min_L,lambda_min_R));
      
      real cmax  = FMAX(FABS(SL), FABS(SR));

      // 3-- Compute the conservative variables: do this by extrapolation
      K_PrimToCons<Phys>(uL, vL, NULL); 
      K_PrimToCons<Phys>(uR, vR, NULL);

      // 4-- Compute the left and right fluxes (wave speed is null)
      K_Flux<Phys,DIR>(fluxL, vL, uL, 0);
      K_Flux<Phys,DIR>(fluxR, vR, uR, 0);
      
      //printf("vR[FR1]=%e, vL[FR1]=%e, SL=%e,SR=%e at i=%i, j=%i, k=%i and DIR=%i\n",vR[Xn],vL[Xn],SL,SR,i,j,k,DIR);

      // 5-- Compute the flux from the left and right states
      if (SL >= 0) {
#pragma unroll
        for (int nv = 0 ; nv < Phys::nvar; nv++) {
          Flux(nv,k,j,i) = fluxL[nv];
        }
      } else if (SR <= 0) {
#pragma unroll
        for (int nv = 0 ; nv < Phys::nvar; nv++) {
          Flux(nv,k,j,i) = fluxR[nv];
        }
      // switch to LFR solver if speeds are small
      } else if (FABS(SL) < SMALL_NUMBER && FABS(SR) < SMALL_NUMBER) {
        //printf("Switch to LFR where velocities are the same at i=%i,j=%i,k=%i\n",i,j,k);
#pragma unroll
        for (int nv = 0 ; nv < Phys::nvar; nv++) {
          Flux(nv,k,j,i) = 0.5*(fluxL[nv] + fluxR[nv]-cmax*(uR[nv]-uL[nv]));
        }
      // switch to HLL if strong shocks
//      } else if (this->haveShockFlattening && ((this->shockFlattening->flagArray(k-koffset,j-joffset,i-ioffset) == FlagShock::Shock) || (this->shockFlattening->flagArray(k,j,i) == FlagShock::Shock))) {
//         //printf("Switch to HLL solver because of shock flattening at i=%i, j=%i, k=%i\n",i,j,k);
//        real dS = SR-SL;
//        if(std::abs(dS) < SMALL_NUMBER) {
//          dS = SMALL_NUMBER;
//          printf("Velocities are the same\n");
//        }
// #pragma unroll
//        for (int nv = 0 ; nv < Phys::nvar; nv++) {
//          Flux(nv,k,j,i) = SL*SR*uR[nv] - SL*SR*uL[nv] + SR*fluxL[nv] - SL*fluxR[nv];
//          Flux(nv,k,j,i) /= dS;
//        }
      } else {
        real dS = SR-SL;
        if(std::abs(dS) < SMALL_NUMBER) {
          dS = SMALL_NUMBER;
          printf("Velocities are the same\n");
        }
        
        // Get U* 
        real FnormL = std::sqrt(EXPAND(vL[FR1]*vL[FR1] , + vL[FR2]*vL[FR2], + vL[FR3]*vL[FR3]));
        real FnormR = std::sqrt(EXPAND(vR[FR1]*vR[FR1] , + vR[FR2]*vR[FR2], + vR[FR3]*vR[FR3]));

        real cos_thetaL = (FnormL <= 1.e-50 ? vL[Xn]/1.e-50 : vL[Xn] / FnormL);
        real cos_thetaR = (FnormR <= 1.e-50 ? vR[Xn]/1.e-50 : vR[Xn] / FnormR);

        real f_paramL = FnormL/vL[ER];
        real f2_paramL = f_paramL*f_paramL;

        real f_paramR = FnormR/vR[ER];
        real f2_paramR = f_paramR*f_paramR;

        real zeta_L = std::sqrt(4.-3.*f2_paramL);

        real zeta_R = std::sqrt(4.-3.*f2_paramR);

        real xiL = 3.+4.*f2_paramL;
        xiL /= 5.+2.*zeta_L;

        real xiR = 3.+4.*f2_paramR;
        xiR /= 5.+2.*zeta_R;

        real betaL = (f2_paramL < 1e-50) ? ZERO_F : (1.5*xiL-0.5)*cos_thetaL/f_paramL;
        real betaR = (f2_paramR < 1e-50) ? ZERO_F : (1.5*xiR-0.5)*cos_thetaR/f_paramR;
        
        real AL = SL*vL[ER] - fluxL[ER];
        real AR = SR*vR[ER] - fluxR[ER];
      
        real BL = SL*vL[Xn] - fluxL[Xn];
        real BR = SR*vR[Xn] - fluxR[Xn];
      
        //printf("vR[ER]=%e, vR[FR1]=%e, fluxR[ER]=%e, vL[ER]=%e, vL[FR1]=%e, fluxL[ER]=%e, AL=%e,AR=%e,BL=%e,BR=%e at i=%i, j=%i, k=%i and DIR=%i\n",vR[ER],vR[Xn],fluxR[ER],vL[ER],vL[Xn],fluxL[ER],AL,AR,BR,BL,i,j,k,DIR);


        real fpL = EXPAND(ZERO_F, + vL[Xt]*vL[Xt], + vL[Xb]*vL[Xb]) ;
        real fpR = EXPAND(ZERO_F, + vR[Xt]*vR[Xt], + vR[Xb]*vR[Xb]) ;

        real eeL = 1e-20*vL[ER] ;
        real eeR = 1e-20*vR[ER] ;

        //if( (fabs(AL)<eeL && fabs(AR)<eeR) || (fabs(fpL)<eeL && fabs(fpR)<eeR) ){
        //if( (fabs(FnormL - vL[ER]) < 1.e-10) && (fabs(FnormR - vR[ER]) < 1.e-10) && ((vL[Xn]/FnormL) <= (vR[Xn]/FnormR))){
        if( (fabs(AL) < eeL) && (fabs(AR) < eeR) && (fabs(BL) < eeL) && (fabs(BR) < eeR)){
#pragma unroll
            for(int nv = 0 ; nv < Phys::nvar; nv++) {
                //printf("Switch to HLL solver because of vacuum like int. states at i=%i, j=%i, k=%i and DIR=%i, FxL/FL=%e, FxR=%e,FR=%e\n",i,j,k,DIR,vL[Xn]/FnormL,vR[Xn],FnormR);
                //printf("Switch to HLL solver because of vacuum like int. states at i=%i, j=%i, k=%i and DIR=%i, AL=%e, AR=%e,BL=%e, BR=%e, f2_paramL=%e, f2_paramR=%e\n",i,j,k,DIR,AL,AR,BL,BR,f2_paramL,f2_paramR);
                Flux(nv,k,j,i) = SL*SR*uR[nv] - SL*SR*uL[nv] + SR*fluxL[nv] - SL*fluxR[nv];
                Flux(nv,k,j,i) /= dS;
                //printf("dS=%e at i=%i\n",dS,i);
            }
        } else {
            real a = AR*SL - AL*SR;
            real b = AL + BL*SR - AR - BR*SL;
            real c = BR - BL;

            real scrh = (b >= ZERO_F) ? -0.5*(b + std::sqrt(b*b - 4.0*a*c)) :  -0.5*(b - std::sqrt(b*b - 4.0*a*c));
            real us   = c/scrh;
            real ps = (AL*us - BL)/(1.0 - us*SL);
                    
            //printf("vR[FR1]=%e, vL[FR1]=%e, AL=%e,AR=%e,BL=%e,BR=%e, scrh=%e, fluxL=%e, fluxR=%e at i=%i, j=%i, k=%i and DIR=%i\n",vR[Xn],vL[Xn],AL,AR,BR,BL,scrh,fluxL[ER],fluxR[ER],i,j,k,DIR);

            EXPAND( usL[Xn] = (SL*(vL[ER] + ps) - vL[Xn])*us/(SL - us);
                    usR[Xn] = (SR*(vR[ER] + ps) - vR[Xn])*us/(SR - us); , 
                    usL[Xt] = vL[Xt]*(SL - betaL)/(SL - us);
                    usR[Xt] = vR[Xt]*(SR - betaR)/(SR - us); ,
                    usL[Xb] = vL[Xb]*(SL - betaL)/(SL - us);
                    usR[Xb] = vR[Xb]*(SR - betaR)/(SR - us); )
        
            usL[ER] = vL[ER] + (usL[Xn]-vL[Xn])/SL ;
            usR[ER] = vR[ER] + (usR[Xn]-vR[Xn])/SR ;

            if (us >= 0.0) {
#pragma unroll
              for(int nv = 0 ; nv < Phys::nvar; nv++) {
                  Flux(nv,k,j,i) = fluxL[nv] + SL*(usL[nv] - uL[nv]);
              }
            } else {
#pragma unroll
              for(int nv = 0 ; nv < Phys::nvar; nv++) {
                  Flux(nv,k,j,i) = fluxR[nv] + SR*(usR[nv] - uR[nv]);
              }
            }    
        }
      }
    
      //printf("Flux(ER,k,j,i)=%e, Flux(FR1,k,j,i)=%e at i=%i, j=%i, k=%i and DIR=%i\n",Flux(ER,k,j,i),Flux(FR1,k,j,i),i,j,k,DIR);
      //printf("vR[ER]=%e, vR[FR1]=%e, fluxR[ER]=%e, fluxR[FR1]=%e, SR=%e, vL[ER]=%e, vL[FR1]=%e, fluxL[ER]=%e, fluxL[FR1]=%e, SL=%e at i=%i, j=%i, k=%i and DIR=%i\n",vR[ER],vR[Xn],fluxR[ER],fluxR[FR1],SR,vL[ER],vL[Xn],fluxL[ER],fluxL[FR1],SL,i,j,k,DIR);


      //6-- Compute maximum wave speed for this sweep
      cMax(k,j,i) = cmax;
    }
  );

  idfx::popRegion();
}

#endif // FLUID_RIEMANNSOLVER_RADSOLVERS_HLLCRAD_HPP_
