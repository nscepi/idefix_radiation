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
#include "radsource.hpp"

// Compute Riemann fluxes from states using HLLC solver
template <typename Phys>
template<const int DIR>
void RiemannSolver<Phys>::HllcRad(IdefixArray4D<real> &Flux) {
  idfx::pushRegion("RiemannSolver::HLLC_Rad");

  constexpr int ioffset = (DIR==IDIR) ? 1 : 0;
  constexpr int joffset = (DIR==JDIR) ? 1 : 0;
  constexpr int koffset = (DIR==KDIR) ? 1 : 0;
  
  IdefixArray4D<real> Vc = this->Vc;
  IdefixArray3D<real> cMax = this->cMax;

  // Required for high order interpolations
  IdefixArray1D<real> dx = this->data->dx[DIR];


  ExtrapolateToFaces<Phys,DIR> extrapol = *this->GetExtrapolator<DIR>();

  // Reduced velocity of light
  real reduced_c = this->reduced_c;

  RadSource &rad_source = *(this->hydro->radsource);

  idefix_for("HLLC_Rad_Kernel",
             data->beg[KDIR],data->end[KDIR]+koffset,
             data->beg[JDIR],data->end[JDIR]+joffset,
             data->beg[IDIR],data->end[IDIR]+ioffset,
    KOKKOS_LAMBDA (int k, int j, int i) {
      // Init the directions (should be in the kernel for proper optimisation by the compilers)
      EXPAND( constexpr int Xn = DIR+MX1;                    ,
              constexpr int Xt = (DIR == IDIR ? MX2 : MX1);  ,
              constexpr int Xb = (DIR == KDIR ? MX2 : MX3);  )
      const int index = ioffset*i + joffset*j + koffset*k;

      // Primitive variables
      real vL[Phys::nvar];
      real vR[Phys::nvar];
      real v[Phys::nvar];
      real voffset[Phys::nvar];

      for(int nv = 0 ; nv < Phys::nvar; nv++) {
        v[nv] = Vc(nv,k,j,i);
        voffset[nv] = Vc(nv,k-koffset,j-joffset,i-ioffset);
      }

      // Conservative variables
      real uL[Phys::nvar], usL[Phys::nvar];
      real uR[Phys::nvar], usR[Phys::nvar];

      // Flux (left and right)
      real fluxL[Phys::nvar];
      real fluxR[Phys::nvar];

      //Wave speeds
      real lambdaL[2];
      real lambdaR[2];
      
      // xi from closure
      real xiL, xiR;

      // 1-- Store the primitive variables on the left, right, and averaged states
      extrapol.ExtrapolatePrimVar(i, j, k, vL, vR);

      // Limit Fr after extrapolation to satisfy Fr<=Er
      K_LimitRadFluxReconstruct(vL,vR,v,voffset);
      //K_LimitRadFlux(vL);
      //K_LimitRadFlux(vR);

      // 2-- Get the wave speed
      K_SpeedsRad(lambdaL,vL,Xn, reduced_c,&xiL);
      K_SpeedsRad(lambdaR,vR,Xn, reduced_c,&xiR);

      real speed_diff = rad_source.LimitSpeedsRad(i,j,k,dx[index]);
 
      real lambda_max_L = FMAX(lambdaL[0],lambdaL[1]);
      real lambda_max_R = FMAX(lambdaR[0],lambdaR[1]);
      real lambda_min_L = FMIN(lambdaL[0],lambdaL[1]);
      real lambda_min_R = FMIN(lambdaR[0],lambdaR[1]);
      
      real SR = FMAX(lambda_max_L,lambda_max_R);
      SR = FMIN(speed_diff,SR);
      real SL = FMIN(lambda_min_L,lambda_min_R);
      SL = FMAX(-speed_diff,SL);

      real cmax  = FMAX(FABS(SL), FABS(SR));

      real dS = SR-SL;

      // 3-- Compute the conservative variables: do this by extrapolation
      K_PrimToCons<Phys>(uL, vL, NULL); 
      K_PrimToCons<Phys>(uR, vR, NULL);

      // 4-- Compute the left and right fluxes (wave speed is null)
      K_Flux<Phys,DIR>(fluxL, vL, uL, reduced_c,xiL);
      K_Flux<Phys,DIR>(fluxR, vR, uR, reduced_c,xiR);
      
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
//       // switch to LFR solver if speeds are small
//      } else if (FABS(SL) < SMALL_NUMBER && FABS(SR) < SMALL_NUMBER) {
      //if (FABS(SL) < SMALL_NUMBER && FABS(SR) < SMALL_NUMBER) {
//      if (FABS(dS) < SMALL_NUMBER) {
//        dS = SMALL_NUMBER;
      //  printf("Switch to LFR where velocities are the same at i=%i,j=%i,k=%i\n",i,j,k);
//#pragma unroll
        //for (int nv = 0 ; nv < Phys::nvar; nv++) {
        //  Flux(nv,k,j,i) = 0.5*(fluxL[nv] + fluxR[nv]-cmax*(uR[nv]-uL[nv]));
        //}
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
        if(std::abs(dS) < SMALL_NUMBER) {
          dS = SMALL_NUMBER;
          //printf("Velocities are the same\n");
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

        real betaL = (f2_paramL < 1.e-100) ? 1.e-100 : (1.5*xiL-0.5)*cos_thetaL/f_paramL;
        betaL *= reduced_c;
        real betaR = (f2_paramR < 1.e-100) ? 1.e-100 : (1.5*xiR-0.5)*cos_thetaR/f_paramR;
        betaR *= reduced_c;

        real AL = SL*vL[ER] - fluxL[ER];
        real AR = SR*vR[ER] - fluxR[ER];
      
        real BL = SL*vL[Xn] - fluxL[Xn];
        BL *= reduced_c;
        real BR = SR*vR[Xn] - fluxR[Xn];
        BR *= reduced_c;

        //printf("vR[ER]=%e, vR[FR1]=%e, fluxR[ER]=%e, vL[ER]=%e, vL[FR1]=%e, fluxL[ER]=%e, AL=%e,AR=%e,BL=%e,BR=%e at i=%i, j=%i, k=%i and DIR=%i\n",vR[ER],vR[Xn],fluxR[ER],vL[ER],vL[Xn],fluxL[ER],AL,AR,BR,BL,i,j,k,DIR);

        //real fpL = EXPAND(ZERO_F, + vL[Xt]*vL[Xt], + vL[Xb]*vL[Xb]) ;
        //real fpR = EXPAND(ZERO_F, + vR[Xt]*vR[Xt], + vR[Xb]*vR[Xb]) ;

        real eeL = 1e-10*vL[ER];
        real eeR = 1e-10*vR[ER];
        real ee = 1.e-10*FMAX(eeL,eeR);
        ee = FMAX(ee,1.e-20);
        //if( (fabs(AL)<(reduced_c*eeL) && fabs(AR)<(reduced_c*eeR)) || (fabs(fpL)<(eeL) && fabs(fpR)<(eeR)) ){
        //if( true ){
        //if( (fabs(FnormL - vL[ER]) < eeL) && (fabs(FnormR - vR[ER]) < eeR) && ((vL[Xn]/FnormL) <= (vR[Xn]/FnormR))){
        if( (fabs(AL) < ee) && (fabs(AR) < ee) && (fabs(BL) < ee) && (fabs(BR) < ee)){
#pragma unroll
            for(int nv = 0 ; nv < Phys::nvar; nv++) {
                //printf("Switch to HLL solver because of vacuum like int. states at i=%i, x=%e, FxL/FL=%e, FxR=%e,FR=%e\n",i,this->data->x[DIR][i],uL[Xn]/FnormL,vR[Xn],FnormR);
                //printf("Switch to HLL solver because of vacuum like int. states at i=%i, j=%i, k=%i and DIR=%i, AL=%e, AR=%e,BL=%e, BR=%e, f2_paramL=%e, f2_paramR=%e\n",i,j,k,DIR,AL,AR,BL,BR,f2_paramL,f2_paramR);
                Flux(nv,k,j,i) = SL*SR*uR[nv] - SL*SR*uL[nv] + SR*fluxL[nv] - SL*fluxR[nv];
                Flux(nv,k,j,i) /= dS;
                //printf("dS=%e at i=%i\n",dS,i);
                if (std::isnan(Flux(nv,k,j,i))){
                  //throw std::runtime_error("Nan in HLL part of solver.");
                  Kokkos::abort("Nan in HLL part of solver.");
                }
            }

        } else {
            real a = AR*SL - AL*SR;
            real b = AL*reduced_c*reduced_c  + BL*SR - AR*reduced_c*reduced_c - BR*SL;
            real c = (BR - BL)*reduced_c*reduced_c;
            real delta;
            // Ensure posivity on delta for stability of the HLLC solver
            if ((b*b - 4.0*a*c < ZERO_F)){
              //printf("delta<0 in HLLC solver! at i=%i,j=%i,k=%i \n",i,j,k);
              delta = ZERO_F;
            } else {
              delta = b*b - 4.0*a*c;
            }

            real scrh = (b >= ZERO_F) ? -0.5*(b + std::sqrt(delta)) :  -0.5*(b - std::sqrt(delta));
            real us   = c/scrh;
            real ps = (AL*us - BL)/(reduced_c*reduced_c - us*SL);
                    
            //printf("vR[FR1]=%e, vL[FR1]=%e, AL=%e,AR=%e,BL=%e,BR=%e, scrh=%e, fluxL=%e, fluxR=%e at i=%i, j=%i, k=%i and DIR=%i\n",vR[Xn],vL[Xn],AL,AR,BR,BL,scrh,fluxL[ER],fluxR[ER],i,j,k,DIR);
            //printf("c=%e at i=%i\n",c,i);

            EXPAND( usL[Xn] = (SL*(vL[ER] + ps) - reduced_c*vL[Xn])*us/(SL - us)/reduced_c;
                    usR[Xn] = (SR*(vR[ER] + ps) - reduced_c*vR[Xn])*us/(SR - us)/reduced_c; , 
                    usL[Xt] = vL[Xt]*(SL - betaL)/(SL - us);
                    usR[Xt] = vR[Xt]*(SR - betaR)/(SR - us); ,
                    usL[Xb] = vL[Xb]*(SL - betaL)/(SL - us);
                    usR[Xb] = vR[Xb]*(SR - betaR)/(SR - us); )                    
        
            usL[ER] = vL[ER] + reduced_c*(usL[Xn]-vL[Xn])/SL ;
            usR[ER] = vR[ER] + reduced_c*(usR[Xn]-vR[Xn])/SR ;

            if (us >= 0.0) {
#pragma unroll
              for(int nv = 0 ; nv < Phys::nvar; nv++) {
                  Flux(nv,k,j,i) = fluxL[nv] + SL*(usL[nv] - uL[nv]);
                  if (std::isnan(Flux(nv,k,j,i))){
                    //printf("SR = %e, SL=%e, vR[ER]=%e, vL[ER]=%e, uR[ER]=%e, uL[ER]=%e, uR[FX1]=%e, uL[FX1]=%e, usR[ER]=%e, usL[ER]=%e,  usR[FX1]=%e, usL[FX1]=%e, ps=%e, us=%e, b*b - 4.0*a*c=%e, AR=%e, AL=%e, BR=%e, BL=%e\n",SR,SL,vR[ER],vL[ER],uR[ER],uL[ER],uR[Xn],uL[Xn],usR[ER],usL[ER],usR[Xn],usL[Xn],ps,us,b*b - 4.0*a*c,AR,AL,BR,BL);
                    //throw std::runtime_error("Nan in HLLC us>0 part of solver.");
                    Kokkos::abort("Nan in HLLC us>0 part of solver.");
                  }
              }
            } else {
#pragma unroll
              for(int nv = 0 ; nv < Phys::nvar; nv++) {
                  Flux(nv,k,j,i) = fluxR[nv] + SR*(usR[nv] - uR[nv]);
                  if (std::isnan(Flux(nv,k,j,i))){
                    //printf("SR = %e, SL=%e, vR[ER]=%e, vL[ER]=%e, uR[ER]=%e, uL[ER]=%e, uR[FX1]=%e, uL[FX1]=%e, usR[ER]=%e, usL[ER]=%e,  usR[FX1]=%e, usL[FX1]=%e, ps=%e, us=%e, b*b - 4.0*a*c=%e, AR=%e, AL=%e, BR=%e, BL=%e\n",SR,SL,vR[ER],vL[ER],uR[ER],uL[ER],uR[Xn],uL[Xn],usR[ER],usL[ER],usR[Xn],usL[Xn],ps,us,b*b - 4.0*a*c,AR,AL,BR,BL);
                    //throw std::runtime_error("Nan in HLLC us<0 part of solver.");
                    Kokkos::abort("Nan in HLLC us<0 part of solver.");
                  }
              }
            }    
        }
      
      }   


      //6-- Compute maximum wave speed for this sweep
      cMax(k,j,i) = cmax;
    }
  );

  idfx::popRegion();
}

#endif // FLUID_RIEMANNSOLVER_RADSOLVERS_HLLCRAD_HPP_
