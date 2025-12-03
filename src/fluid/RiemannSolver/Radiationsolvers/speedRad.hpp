#ifndef FLUID_RIEMANNSOLVER_RADSOLVERS_SPEEDRAD_HPP_
#define FLUID_RIEMANNSOLVER_RADSOLVERS_SPEEDRAD_HPP_

#include <string>
#include <tuple>
#include "idefix.hpp"
#include "input.hpp"


KOKKOS_INLINE_FUNCTION void K_SpeedsRad(real lambda[], const real *KOKKOS_RESTRICT V, int Xn, real reduced_c) {
    
    real Fnorm = std::sqrt(EXPAND(V[FR1]*V[FR1] , + V[FR2]*V[FR2], + V[FR3]*V[FR3]));

    real f_param = Fnorm/V[ER];

    if (f_param < 1.e-50) {
        lambda[1] = ONE_F/std::sqrt(3.);
        lambda[0] = -lambda[1];
    } else {
        real f2_param = f_param*f_param;
      
        real cos_theta = V[Xn] / Fnorm;
      
        // Eq. 85 of Melon Fuksman & Mignone 2019  
        real zeta_1 = FABS(4.-3.*f2_param);
        real zeta_2 = std::sqrt(zeta_1);
        real zeta_3 = 2.*(zeta_1-zeta_2)/3.;
        real zeta_4 = 2.*cos_theta*cos_theta*(2.-f2_param-zeta_2);
        real zeta = std::sqrt(FABS(zeta_3 + zeta_4));

        // Eq. 82-84 of Melon Fuksman & Mignone 2019  
        lambda[0] = f_param*cos_theta-zeta;
        lambda[0] /= zeta_2;
        // We never use the intermediate speed so we don't compute it
        lambda[1] = f_param*cos_theta+zeta;
        lambda[1] /= zeta_2;

    }
    
    lambda[0] *= reduced_c;
    lambda[1] *= reduced_c;
    

    return;
}


#endif //FLUID_RIEMANNSOLVER_RADSOLVERS_SPEEDRAD_HPP_
