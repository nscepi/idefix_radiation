#ifndef FLUID_RIEMANNSOLVER_RADSOLVERS_SPEEDRAD_HPP_
#define FLUID_RIEMANNSOLVER_RADSOLVERS_SPEEDRAD_HPP_

#include <string>
#include <tuple>
#include "idefix.hpp"
#include "input.hpp"


KOKKOS_INLINE_FUNCTION void K_SpeedsRad(real lambda[], const real *KOKKOS_RESTRICT V, int Xn, real reduced_c, real *xi) {
    
    real Fnorm = Kokkos::sqrt(EXPAND(V[FR1]*V[FR1] , + V[FR2]*V[FR2], + V[FR3]*V[FR3]));
    real f = (V[ER] < 1.e-50 ? Fnorm/(1.e-50) : Fnorm/V[ER]);
    real f2 = f*f;
    
    real cos_theta = V[Xn] / Fnorm;

    // Eq. 85 of Melon Fuksman & Mignone 2019  
    real zeta_1 = 4.-3.*f2;  // I removed an absolute value, see if it still work 
    real zeta_2 = Kokkos::sqrt(4.-3.*f2);
    real zeta_3 = 2.*(zeta_1-zeta_2)/3.;
    real zeta_4 = 2.*cos_theta*cos_theta*(2.-f2-zeta_2);
    real zeta = Kokkos::sqrt(zeta_3 + zeta_4); // I removed an absolute value inside the sqrt, see if it still work 

    if (f < 1.e-50) {
        lambda[1] = 0.5773502691896258; // 1/sqrt(3)
        lambda[0] = -lambda[1];
    } else {      
        // Eq. 82-84 of Melon Fuksman & Mignone 2019  
        lambda[0] = f*cos_theta-zeta;
        lambda[0] /= zeta_2;
        // We never use the intermediate speed so we don't compute it
        lambda[1] = f*cos_theta+zeta;
        lambda[1] /= zeta_2;
    }
    lambda[0] *= reduced_c;
    lambda[1] *= reduced_c;
    
    *xi  = 3.+4.*f2;
    *xi /= 5.+2.*zeta_2;  

    return;
}


#endif //FLUID_RIEMANNSOLVER_RADSOLVERS_SPEEDRAD_HPP_
