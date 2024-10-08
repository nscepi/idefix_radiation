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
#include "lookupTable.hpp"

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

  const real C_c = idfx::units.c;
  real C_ar = idfx::units.ar;

  real unit_velocity = idfx::units.velocity;
  real unit_length = idfx::units.length;
  real unit_density = idfx::units.density;
  real KELVIN = idfx::units.Kelvin;
  real unit_time = unit_length/unit_velocity;
  real unit_energy = unit_density*unit_velocity*unit_velocity;
  // Max iteration for fixed-point solver
  int MAX_ITER = 100;
  // Tolerance on ER and ENG for fixed-point solver
  real tol = 1.e-10;

  EquationOfState eos = *(this->eos);

  auto k_p = this->kappa_planck;
  auto k_r = this->kappa_ross;
  
  idefix_for("RadSource",0,data->np_tot[KDIR],0,data->np_tot[JDIR],0,data->np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
  
      real Etot = UcGas(ENG,k,j,i)+UcRad(ER,k,j,i)/reduced_c;
      real m1tot = UcGas(MX1,k,j,i)+UcRad(FR1,k,j,i)*unit_velocity/(reduced_c*C_c);
      real m2tot = UcGas(MX2,k,j,i)+UcRad(FR2,k,j,i)*unit_velocity/(reduced_c*C_c);
      real m3tot = UcGas(MX3,k,j,i)+UcRad(FR3,k,j,i)*unit_velocity/(reduced_c*C_c);
      
      real Er_hyp = UcRad(ER,k,j,i);
      real Fr1_hyp = UcRad(FR1,k,j,i);
      real Fr2_hyp = UcRad(FR2,k,j,i);
      real Fr3_hyp = UcRad(FR3,k,j,i);

      real Egas_hyp = UcGas(ENG,k,j,i);
      real m1gas_hyp = UcGas(MX1,k,j,i);
      real m2gas_hyp = UcGas(MX2,k,j,i);
      real m3gas_hyp = UcGas(MX3,k,j,i);

      real URad[RadiationPhysics::nvar];
      real VRad[RadiationPhysics::nvar];
      real Er_old, Fnorm_old, Egas_old, Mnorm_old;

      real UGas[DefaultPhysics::nvar];
      real VGas[DefaultPhysics::nvar];
      
      real kappa_p;
      real kappa_r;

      // Constant to stabilize implicit step
      real s = 1.;

      if (kappa_type == Type::constant){
        kappa_p = kappa_0;
        kappa_r = kappa_0;
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
      real Mnorm = std::sqrt(EXPAND(UGas[MX1]*UGas[MX1] , + UGas[MX2]*UGas[MX2], + UGas[MX3]*UGas[MX3]));

      // If E_gas > E_rad iterate on radiative field
      //if (UGas[ENG] > URad[ER]/reduced_c){
      if (true){
        while (((err1>tol) || (err2>tol)) && (count < MAX_ITER)){

          Er_old = URad[ER];
          Fnorm_old = Fnorm;
        
          real T = VGas[PRS]/(VGas[RHO])*KELVIN*mu;
          real logT = std::log10(T);
          if (kappa_type == Type::kramers){
            kappa_p = kappa_0*(VGas[RHO]*unit_density/rho_0)*std::pow(T/T_0,-3.5);
            kappa_r = kappa_p;
          } else if (kappa_type == Type::usertable){
            kappa_p = k_p.Get(&logT);
            kappa_r = k_r.Get(&logT);
            //printf("T=%e,kappa_p=%e\n",T,kappa);
          }
          real kk_red = s*reduced_c * C_c * dt * unit_time * kappa_p * VGas[RHO]*unit_density;
          real xx_red = s*reduced_c * C_c * dt * unit_time * (xi_rad + kappa_r) * VGas[RHO]*unit_density;

          URad[ER] = Er_hyp +  kk_red*C_ar*std::pow(T,4.)/unit_energy;
          URad[ER] /= 1. + kk_red;
          EXPAND( URad[FR1] = Fr1_hyp/(1.+xx_red);,
                  URad[FR2] = Fr2_hyp/(1.+xx_red);,
                  URad[FR3] = Fr3_hyp/(1.+xx_red);)
          Fnorm = std::sqrt(EXPAND(URad[FR1]*URad[FR1] , + URad[FR2]*URad[FR2], + URad[FR3]*URad[FR3]));

          // Change value of s if UGas <= 0
          if ((Etot - URad[ER]/reduced_c)<=ZERO_F) {
            printf("UGas[ENG]=%e URad[ER]*c/c_red=%e Etot=%e at i=%i j=%i and k=%i at iteration %i with Egas=%e and Erad*c/c_red=%e at iteration 0\n",UGas[ENG],URad[ER]/reduced_c,Etot,i,j,k,count,UcGas(ENG,k,j,i),UcRad(ER,k,j,i)/reduced_c);
            throw std::runtime_error("ENG=0 in Radsource");
            //URad[ER] = (Etot -UGas[ENG])*reduced_c;
            //s *= 0.1;
            //continue;
          } else {
            UGas[ENG] = Etot - URad[ER]/reduced_c;
          }

          EXPAND( UGas[MX1] = m1tot - URad[FR1]*unit_velocity/(reduced_c*C_c);,
                  UGas[MX2] = m2tot - URad[FR2]*unit_velocity/(reduced_c*C_c);,
                  UGas[MX3] = m3tot - URad[FR3]*unit_velocity/(reduced_c*C_c);)

          K_ConsToPrim<DefaultPhysics>(VGas, UGas, &eos);

          err1 = std::abs(1.-URad[ER]/Er_old);
          err2 = std::abs(1.-Fnorm/Fnorm_old);
          count += 1;
        } 
      
      // If E_rad > E_gas iterate on hydro field
      } else {
        while (((err1>tol) || (err2>tol)) && (count < MAX_ITER)){

          //Er_old = URad[ER];
          //Fnorm_old = Fnorm;
          Egas_old = UGas[ENG];
          Mnorm_old = Mnorm;
        
          real T = VGas[PRS]/(VGas[RHO])*KELVIN*mu;
          real logT = std::log10(T);
          if (kappa_type == Type::kramers){
            kappa_p = kappa_0*(VGas[RHO]*unit_density/rho_0)*std::pow(T/T_0,-3.5);
            kappa_r = kappa_p;
          } else if (kappa_type == Type::usertable){
            kappa_p = k_p.Get(&logT);
            kappa_r = k_r.Get(&logT);
            //printf("T=%e,kappa_p=%e\n",T,kappa);
          }

          // Why do we have reduced_c here? Does not work if I remove it and it is used in PLUTO...
          real kk =  reduced_c * C_c * dt * unit_time * kappa_p * VGas[RHO]*unit_density;
          real xx =  reduced_c * C_c * dt * unit_time * (xi_rad + kappa_r) * VGas[RHO]*unit_density;

          // Stop if UGas <= 0
          if ((Egas_hyp +  kk*(URad[ER]-C_ar*std::pow(T,4)/unit_energy))<=ZERO_F) {
            printf("UGas[ENG]=%e URad[ER]*c/c_red=%e Etot=%e at i=%i j=%i and k=%i at iteration %i with Egas=%e and Erad*c/c_red=%e at iteration 0\n",UGas[ENG],URad[ER]/reduced_c,Etot,i,j,k,count,UcGas(ENG,k,j,i),UcRad(ER,k,j,i)/reduced_c);
            throw std::runtime_error("EGas=0 in Radsource");
            UGas[ENG] = 1.e-6;
          } else {
            UGas[ENG] = Egas_hyp +  kk*(URad[ER]-C_ar*std::pow(T,4)/unit_energy);
          }

          EXPAND( UGas[MX1] = m1gas_hyp + URad[FR1]*xx;,
                  UGas[MX2] = m2gas_hyp + URad[FR2]*xx;,
                  UGas[MX3] = m3gas_hyp + URad[FR3]*xx;)
          Mnorm = std::sqrt(EXPAND(UGas[MX1]*UGas[MX1] , + UGas[MX2]*UGas[MX2], + UGas[MX3]*UGas[MX3]));

          URad[ER] = (Etot - UGas[ENG])*reduced_c;

          // Stop if URad <= 0
          if (URad[ER]<=ZERO_F) {
            printf("UGas[ENG]=%e URad[ER]*c/c_red=%e Etot=%e  at i=%i j=%i and k=%i at iteration %i with Egas=%e and Erad*c/c_red=%e at iteration 0\n",UGas[ENG],URad[ER]/reduced_c,Etot,i,j,k,count,UcGas(ENG,k,j,i),UcRad(ER,k,j,i)/reduced_c);
            throw std::runtime_error("ERad=0 in Radsource");
          }

          EXPAND( URad[FR1] = (m1tot - UGas[MX1])*reduced_c*C_c/unit_velocity;,
                  URad[FR2] = (m2tot - UGas[MX2])*reduced_c*C_c/unit_velocity;,
                  URad[FR3] = (m3tot - UGas[MX3])*reduced_c*C_c/unit_velocity;)

          K_ConsToPrim<DefaultPhysics>(VGas, UGas, &eos);

          err1 = std::abs(1.-UGas[ENG]/Egas_old);
          err2 = std::abs(1.-Mnorm/Mnorm_old);
          count += 1;
        }
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

real RadSource::Limit_speeds_Rad(int i, int j, int k, real dx) const{
  auto VcGas = this->VcGas;
  real kappa_0 = this->kappa_0;
  auto kappa_type = this->kappa_type;
  real xi_rad = this->xi_rad;
  real mu =this->mu;
  real KELVIN = idfx::units.Kelvin;
  real kappa;
  if (kappa_type == Type::constant){
    kappa = kappa_0;
  } else if (kappa_type == Type::kramers){
    real T = VcGas(PRS,k,j,i)/(VcGas(RHO,k,j,i))*KELVIN*mu;
    kappa = kappa_0*std::pow(VcGas(RHO,k,j,i)*unit_density/rho_0,2.)*std::pow(T/T_0,-3.5);
  }
  real tau = VcGas(RHO,k,j,i)*idfx::units.density*(kappa+xi_rad)*dx*idfx::units.length;

  return 4./(3.*tau)*this->reduced_c*idfx::units.c/idfx::units.velocity;
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
    case Type::usertable:
      idfx::cout << "kappa from table provided by user in source term integration";
      break;
  }
}
