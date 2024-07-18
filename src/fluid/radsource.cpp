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
  
  const Type kappa_type = this->kappa_type;
  real kappa_0 = this->kappa_0;
  real rho_0 = this->rho_0;
  real T_0 = this->T_0;
  real xi_rad = this->xi_rad;
  real reduced_c = this->reduced_c;
  real gamma = this->gamma;
  real mu = this->mu;

  const real C_c = this->C_c;
  real C_ar = this->C_ar;

  real unit_velocity = this->unit_velocity;
  real unit_length = this->unit_length;
  real unit_density = this->unit_density;
  real KELVIN = this->Kelvin;
  real unit_time = unit_length/unit_velocity;
  real unit_energy = unit_density*unit_velocity*unit_velocity;
  // Max iteration for fixed-point solver
  int MAX_ITER = 100;
  // Tolerance on ER and ENG for fixed-point solver
  real tol = 1.e-10;

  EquationOfState eos = *(this->eos);

  idefix_for("RadSource",0,data->np_tot[KDIR],0,data->np_tot[JDIR],0,data->np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
  
      real Etot = UcGas(ENG,k,j,i)+UcRad(ER,k,j,i)*C_c/(reduced_c*unit_velocity);
      real m1tot = UcGas(MX1,k,j,i)+UcRad(FR1,k,j,i)/(reduced_c*unit_velocity);
      real m2tot = UcGas(MX2,k,j,i)+UcRad(FR2,k,j,i)/(reduced_c*unit_velocity);
      real m3tot = UcGas(MX3,k,j,i)+UcRad(FR3,k,j,i)/(reduced_c*unit_velocity);
      
      real Er_hyp = UcRad(ER,k,j,i);
      real Fr1_hyp = UcRad(FR1,k,j,i);
      real Fr2_hyp = UcRad(FR2,k,j,i);
      real Fr3_hyp = UcRad(FR3,k,j,i);

      real URad[RadiationPhysics::nvar];
      real VRad[RadiationPhysics::nvar];
      real Er_old, Fnorm_old;

      real UGas[DefaultPhysics::nvar];
      real VGas[DefaultPhysics::nvar];
      
      real kappa;
      if (kappa_type == Type::constant){
        kappa = kappa_0;
      }

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
        if (kappa_type == Type::kramers){
          kappa = kappa_0*(VGas[RHO]*unit_density/rho_0)*std::pow(T/T_0,-3.5);
        }
        real kk_red = reduced_c * unit_velocity * dt * unit_time * kappa * VGas[RHO]*unit_density;
        real xx_red = reduced_c * unit_velocity * dt * unit_time * (xi_rad + kappa) * VGas[RHO]*unit_density;

        URad[ER] = Er_hyp +  kk_red*C_ar*std::pow(T,4)/unit_energy;
        URad[ER] /= 1. + kk_red;
        EXPAND( URad[FR1] = Fr1_hyp/(1.+xx_red);,
                URad[FR2] = Fr2_hyp/(1.+xx_red);,
                URad[FR3] = Fr3_hyp/(1.+xx_red);)

        UGas[ENG] = Etot - URad[ER]*C_c/(reduced_c*unit_velocity);

        // Fix if UGas < 0
        if (UGas[ENG]<ZERO_F) {
          printf("Gas Energy is <0 at i=%i, j=%i, k=%i\n",i,j,k);
          UGas[ENG] = (&eos)->GetInternalEnergy(SMALL_PRESSURE_FIX,VGas[RHO]);
        }

        EXPAND( UGas[MX1] = m1tot - URad[FR1]/(reduced_c*unit_velocity);,
                UGas[MX2] = m2tot - URad[FR2]/(reduced_c*unit_velocity);,
                UGas[MX3] = m3tot - URad[FR3]/(reduced_c*unit_velocity);)

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
  real kappa_0 = this->kappa_0;
  real xi_rad = this->xi_rad;
  real KELVIN = this->Kelvin;
  real mu =1.;
  real kappa;
  if (kappa_type == Type::constant){
    kappa = kappa_0;
  } else if (kappa_type == Type::kramers){
    real T = VcGas(PRS,k,j,i)/(VcGas(RHO,k,j,i))*KELVIN*mu;
    kappa = kappa_0*std::pow(VcGas(RHO,k,j,i)*unit_density/rho_0,2.)*std::pow(T/T_0,-3.5);
  }
  real tau = VcGas(RHO,k,j,i)*this->unit_density*(kappa+xi_rad)*dx*this->unit_length;

  return 4./(3.*tau)*this->reduced_c*this->C_c/this->unit_velocity;
}

void RadSource::ShowConfig() {
  idfx::cout << "RadSource: Using ";
  switch(kappa_type) {
    case Type::constant:
      idfx::cout << "constant kappa in source term integration";
      break;
    case Type::kramers:
      idfx::cout << "kappa in kramers form in source term integration";
      break;
  }
}
