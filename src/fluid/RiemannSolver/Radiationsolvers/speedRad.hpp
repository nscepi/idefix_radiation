#ifndef FLUID_RIEMANNSOLVER_RADSOLVERS_SPEEDRAD_HPP_
#define FLUID_RIEMANNSOLVER_RADSOLVERS_SPEEDRAD_HPP_

#include <string>
#include <tuple>
#include "idefix.hpp"
#include "input.hpp"


KOKKOS_INLINE_FUNCTION void K_speeds_Rad(real lambda[], real f_param, real f2_param, real cos_theta) {
    
    real zeta_1 = 4.-3.*f2_param;
    real zeta_2 = std::sqrt(zeta_1);
    real zeta_3 = 2.*(zeta_1-zeta_2)/3.;
    real zeta_4 = 2.*cos_theta*cos_theta*(2.-f2_param-zeta_2);
    real zeta = std::sqrt(zeta_3 + zeta_4);

    real epsilon = 3.+4.*f2_param;
    epsilon /= 5.+2.*zeta_2;

    lambda[0] = f_param*cos_theta-zeta;
    lambda[0] /= zeta_2;
    lambda[1] = (3*epsilon-1.)*cos_theta;
    lambda[1] /= 2.*f_param;
    lambda[2] = f_param*cos_theta+zeta;
    lambda[2] /= zeta_2;

    return;
}


#endif //FLUID_RIEMANNSOLVER_RADSOLVERS_SPEEDRAD_HPP_
