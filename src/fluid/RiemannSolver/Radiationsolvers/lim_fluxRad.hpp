#ifndef FLUID_RIEMANNSOLVER_RADSOLVERS_LIMFLUXRAD_HPP_
#define FLUID_RIEMANNSOLVER_RADSOLVERS_LIMFLUXRAD_HPP_

#include <string>
#include <tuple>
#include "idefix.hpp"
#include "input.hpp"


KOKKOS_INLINE_FUNCTION void K_LimitRadFlux(real  V[]) {

real Fnorm = std::sqrt(EXPAND(V[FR1]*V[FR1] , + V[FR2]*V[FR2], + V[FR3]*V[FR3]));
       
//real small_factor = ONE_F-2.e-16;
real small_factor = 1.e-15;
if (Fnorm > V[ER]) {
    //printf("Limit Flux Fnorm/Er=%e \n",Fnorm/V[ER]);   
    EXPAND( V[FR1] *= (Fnorm <= 1.e-50 ? V[ER]/1.e-50 : (V[ER]/Fnorm)-small_factor);, 
            V[FR2] *= (Fnorm <= 1.e-50 ? V[ER]/1.e-50 : (V[ER]/Fnorm)-small_factor);,
            V[FR3] *= (Fnorm <= 1.e-50 ? V[ER]/1.e-50 : (V[ER]/Fnorm)-small_factor);)
}

return;
}

KOKKOS_INLINE_FUNCTION void K_LimitRadFluxFlatten(real  vL[], real vR[], real Vc[], real VOffset[]) {

real FnormL = std::sqrt(EXPAND(vL[FR1]*vL[FR1] , + vL[FR2]*vL[FR2], + vL[FR3]*vL[FR3]));
real FnormR = std::sqrt(EXPAND(vR[FR1]*vR[FR1] , + vR[FR2]*vR[FR2], + vR[FR3]*vR[FR3]));

real Fnorm = std::sqrt(EXPAND(Vc[FR1]*Vc[FR1] , + Vc[FR2]*Vc[FR2], + Vc[FR3]*Vc[FR3]));
if (Fnorm > Vc[ER]) {
    //printf("Limit Flux Fnorm/Er=%e \n",Fnorm/V[ER]);   
    EXPAND( Vc[FR1] *= (Fnorm <= 1.e-50 ? Vc[ER]/1.e-50 : Vc[ER]/Fnorm);, 
            Vc[FR2] *= (Fnorm <= 1.e-50 ? Vc[ER]/1.e-50 : Vc[ER]/Fnorm);,
            Vc[FR3] *= (Fnorm <= 1.e-50 ? Vc[ER]/1.e-50 : Vc[ER]/Fnorm);)
}

real FnormOffset = std::sqrt(EXPAND(VOffset[FR1]*VOffset[FR1] , + VOffset[FR2]*VOffset[FR2], + VOffset[FR3]*VOffset[FR3]));
if (FnormOffset > VOffset[ER]) {
    //printf("Limit Flux Fnorm/Er=%e \n",Fnorm/V[ER]);   
    EXPAND( VOffset[FR1] *= (FnormOffset <= 1.e-50 ? VOffset[ER]/1.e-50 : VOffset[ER]/FnormOffset);, 
            VOffset[FR2] *= (FnormOffset <= 1.e-50 ? VOffset[ER]/1.e-50 : VOffset[ER]/FnormOffset);,
            VOffset[FR3] *= (FnormOffset <= 1.e-50 ? VOffset[ER]/1.e-50 : VOffset[ER]/FnormOffset);)
}

if ((FnormL > vL[ER]) || (FnormR > vR[ER])) {
    //printf("Limit Flux Fnorm/Er=%e \n",Fnorm/V[ER]);   
    EXPAND( vL[ER] = VOffset[ER];
            vR[ER] = Vc[ER];
            vL[FR1] = VOffset[FR1]; 
            vR[FR1] = Vc[FR1];, 
            vL[FR2] = VOffset[FR2];
            vR[FR2] = Vc[FR2];,
            vL[FR3] = VOffset[FR3];
            vR[FR3] = Vc[FR3];)
}

return;
}

KOKKOS_INLINE_FUNCTION void K_LimitRadFluxReconstruct(real  vL[], real vR[], real Vc[], real VOffset[]) {

real FnormL = std::sqrt(EXPAND(vL[FR1]*vL[FR1] , + vL[FR2]*vL[FR2], + vL[FR3]*vL[FR3]));
real FnormR = std::sqrt(EXPAND(vR[FR1]*vR[FR1] , + vR[FR2]*vR[FR2], + vR[FR3]*vR[FR3]));
real FnormOffset = std::sqrt(EXPAND(VOffset[FR1]*VOffset[FR1] , + VOffset[FR2]*VOffset[FR2], + VOffset[FR3]*VOffset[FR3]));
real Fnorm = std::sqrt(EXPAND(Vc[FR1]*Vc[FR1] , + Vc[FR2]*Vc[FR2], + Vc[FR3]*Vc[FR3]));

real ratiocellOffset = FnormOffset/VOffset[ER];
real ratiocell = Fnorm/Vc[ER];
real ratioL = FnormL/vL[ER];
real ratioR = FnormR/vR[ER];

if ((FnormL > vL[ER]) || (ratiocellOffset > ratioL)){
    EXPAND( vL[FR1] = VOffset[FR1]*vL[ER]/VOffset[ER];, 
            vL[FR2] = VOffset[FR2]*vL[ER]/VOffset[ER];,
            vL[FR3] = VOffset[FR3]*vL[ER]/VOffset[ER];)
}
if ((FnormR > vR[ER])|| (ratiocell > ratioR)) {
    EXPAND( vR[FR1] = Vc[FR1]*vR[ER]/Vc[ER];, 
            vR[FR2] = Vc[FR2]*vR[ER]/Vc[ER];,
            vR[FR3] = Vc[FR3]*vR[ER]/Vc[ER];)
}


return;
}


#endif //FLUID_RIEMANNSOLVER_RADSOLVERS_LIMFLUXRAD_HPP_
