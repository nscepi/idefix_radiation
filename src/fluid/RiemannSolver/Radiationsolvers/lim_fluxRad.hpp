#ifndef FLUID_RIEMANNSOLVER_RADSOLVERS_LIMFLUXRAD_HPP_
#define FLUID_RIEMANNSOLVER_RADSOLVERS_LIMFLUXRAD_HPP_

#include <string>
#include <tuple>
#include "idefix.hpp"
#include "input.hpp"


KOKKOS_INLINE_FUNCTION void K_limit_RadFlux(real  U[]) {

real reduced_c = 1.; // to change

real Fnorm = std::sqrt(EXPAND(U[FR1]*U[FR1] , + U[FR2]*U[FR2], + U[FR3]*U[FR3]));
      
if (Fnorm > reduced_c*U[ER]) {
    EXPAND( U[FR1] *= (Fnorm <= 1.e-50 ? reduced_c*U[ER]/1.e-50 : reduced_c*U[ER]/Fnorm);, 
            U[FR2] *= (Fnorm <= 1.e-50 ? reduced_c*U[ER]/1.e-50 : reduced_c*U[ER]/Fnorm);,
            U[FR3] *= (Fnorm <= 1.e-50 ? reduced_c*U[ER]/1.e-50 : reduced_c*U[ER]/Fnorm);)   
}

return;
}


#endif //FLUID_RIEMANNSOLVER_RADSOLVERS_LIMFLUXRAD_HPP_
