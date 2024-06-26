// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************
#include "../idefix.hpp"
#include "radsource.hpp"
#include "physics.hpp"
#include "units.hpp"

void RadSource::AddRadSource(const real dt) {
  idfx::pushRegion("RadSource::AddRadSource");

  auto UcGas = this->UcGas;
  auto VcGas = this->VcGas;
  auto UcRad = this->UcRad;
  auto VcRad = this->VcRad;
  auto InvDt = this->InvDt;
  
  const Type type = this->type;
  real kappa_rad = this->kappa_rad;
  real xi_rad = this->xi_rad;
  real reduced_c = this->reduced_c;
  real gamma = this->gamma;
  
  const real C_c = this->C_c;
  real C_ar = this->C_ar;

  real unit_velocity = this->unit_velocity;
  real unit_length = this->unit_length;
  real unit_density = this->unit_density;
  real KELVIN = this->Kelvin;
  real unit_time = unit_length/unit_velocity;
  real unit_energy = unit_density*unit_velocity*unit_velocity;
  real mu = 1.;
  // Max iteration for fixed-point solver
  int MAX_ITER = 100;
  // Tolerance on ER and ENG for fixed-point solver
  real tol = 1.e-10;

  EquationOfState eos = *(this->eos);

  idefix_for("RadSource",0,data->np_tot[KDIR],0,data->np_tot[JDIR],0,data->np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
  
      real Etot = UcGas(ENG,k,j,i)+UcRad(ER,k,j,i)/reduced_c;
      real m1tot = UcGas(MX1,k,j,i)+UcRad(FR1,k,j,i)/(reduced_c*C_c);
      real m2tot = UcGas(MX2,k,j,i)+UcRad(FR2,k,j,i)/(reduced_c*C_c);
      real m3tot = UcGas(MX3,k,j,i)+UcRad(FR3,k,j,i)/(reduced_c*C_c);
      
      real Er_hyp = UcRad(ER,k,j,i);
      real Fr1_hyp = UcRad(FR1,k,j,i);
      real Fr2_hyp = UcRad(FR2,k,j,i);
      real Fr3_hyp = UcRad(FR3,k,j,i);

      real URad[RadiationPhysics::nvar];
      real VRad[RadiationPhysics::nvar];
      real Er_old, Fnorm_old;

      real UGas[DefaultPhysics::nvar];
      real VGas[DefaultPhysics::nvar];
      
      for(int nv = 0 ; nv < RadiationPhysics::nvar ; nv++) {
        URad[nv] = UcRad(nv,k,j,i);
        VRad[nv] = VcRad(nv,k,j,i);
      }

      for(int nv = 0 ; nv < DefaultPhysics::nvar ; nv++) {
        UGas[nv] = UcGas(nv,k,j,i);
        VGas[nv] = VcGas(nv,k,j,i);
      }
    
      real err1= 1.;
      real err2= 1.;
      int count = 0;

      real Fnorm = std::sqrt(EXPAND(URad[FR1]*URad[FR1] , + URad[FR2]*URad[FR2], + URad[FR3]*URad[FR3]));

      while (((err1>tol) || (err2>tol)) && (count < MAX_ITER)){

        Er_old = URad[ER];
        Fnorm_old = Fnorm;
        
        real T = VGas[PRS]/(VGas[RHO])*KELVIN*mu;
        real kk_red = reduced_c * C_c * dt * unit_time * kappa_rad * VGas[RHO]*unit_density;
        real xx_red = reduced_c * C_c * dt * unit_time * (xi_rad + kappa_rad) * VGas[RHO]*unit_density;

        URad[ER] = Er_hyp +  kk_red*C_ar*std::pow(T,4)/unit_energy;
        URad[ER] /= 1. + kk_red;
        EXPAND( URad[FR1] = Fr1_hyp/(1.+xx_red);,
                URad[FR2] = Fr2_hyp/(1.+xx_red);,
                URad[FR3] = Fr3_hyp/(1.+xx_red);)

        UGas[ENG] = Etot - URad[ER]/reduced_c;
        
        // Fix if UGas < 0
        if (UGas[ENG]<ZERO_F) {
          printf("Gas Energy is <0\n");
          UGas[ENG] = (&eos)->GetInternalEnergy(SMALL_PRESSURE_FIX,VGas[RHO]);
        }

        EXPAND( UGas[MX1] = m1tot - URad[FR1]/(reduced_c*C_c);,
                UGas[MX2] = m2tot - URad[FR2]/(reduced_c*C_c);,
                UGas[MX3] = m3tot - URad[FR3]/(reduced_c*C_c);)

        K_ConsToPrim<DefaultPhysics>(VGas, UGas, &eos);

        err1 = std::abs(1.-URad[ER]/Er_old);
        err2 = std::abs(1.-Fnorm/Fnorm_old);
        count += 1;
      }

      for(int nv = 0 ; nv < RadiationPhysics::nvar ; nv++) {
        UcRad(nv,k,j,i) = URad[nv];
        VcRad(nv,k,j,i) = URad[nv];
      }

      for(int nv = 0 ; nv < DefaultPhysics::nvar ; nv++) {
        UcGas(nv,k,j,i) = UGas[nv];
        VcGas(nv,k,j,i) = VGas[nv];
      }
    });

  idfx::popRegion();
}

real RadSource::Limit_speeds_Rad(int i, int j, int k, real dx) {
  auto VcGas = this->VcGas;
  real kappa_rad = this->kappa_rad;
  real xi_rad = this->xi_rad;
  real tau = VcGas(RHO,k,j,i)*this->unit_density*(kappa_rad+xi_rad)*dx*this->unit_length;

  return 4./(3.*tau);
}

void RadSource::ShowConfig() {
  idfx::cout << "RadSource: Using ";
  switch(type) {
    case Type::Tconst:
      idfx::cout << "constant Temperature in source term integration";
      break;
    case Type::Tvar:
      idfx::cout << "Temperature solved with additional equation";
      break;
  }
}
