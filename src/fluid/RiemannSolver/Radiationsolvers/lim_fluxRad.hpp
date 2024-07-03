#ifndef FLUID_RIEMANNSOLVER_RADSOLVERS_LIMFLUXRAD_HPP_
#define FLUID_RIEMANNSOLVER_RADSOLVERS_LIMFLUXRAD_HPP_

#include <string>
#include <tuple>
#include "idefix.hpp"
#include "input.hpp"


KOKKOS_INLINE_FUNCTION void K_limit_RadFlux(real  V[]) {

real Fnorm = std::sqrt(EXPAND(V[FR1]*V[FR1] , + V[FR2]*V[FR2], + V[FR3]*V[FR3]));
      
if (Fnorm > V[ER]) {
    //printf("Limit Flux Fnorm/Er=%e \n",Fnorm/V[ER]);   
    EXPAND( V[FR1] *= (Fnorm <= 1.e-50 ? V[ER]/1.e-50 : V[ER]/Fnorm);, 
            V[FR2] *= (Fnorm <= 1.e-50 ? V[ER]/1.e-50 : V[ER]/Fnorm);,
            V[FR3] *= (Fnorm <= 1.e-50 ? V[ER]/1.e-50 : V[ER]/Fnorm);)
}

return;
}


#endif //FLUID_RIEMANNSOLVER_RADSOLVERS_LIMFLUXRAD_HPP_
