// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************
#include "../idefix.hpp"
#include "radsource.hpp"
#include "physics.hpp"


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
  real C_kb = 1.38e-16;
  real C_mp = 1.6726e-24;
  real C_ar = 7.5646e-15;
  real C_c = 2.99e10;
  real mu = 1.;
  real KELVIN = C_kb/(C_mp*mu);
  real tol = 1.e-10;
  real Etot;

  real unit_velocity = this->unit_velocity;
  real unit_length = this->unit_length;
  real unit_mass = this->unit_mass;
  real unit_time = unit_length/unit_velocity;
  real unit_vol = std::pow(unit_length,3);
  real unit_density = unit_mass/std::pow(unit_length,3);
  real unit_energy = unit_mass/(std::pow(unit_time,2)*unit_length);
  real unit_temp = std::pow(unit_energy/C_ar,0.25);

  int MAX_ITER = 100;
  this->count_max = 0;

  EquationOfState eos = *(this->eos);

  idefix_for("RadSource",0,data->np_tot[KDIR],0,data->np_tot[JDIR],0,data->np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
   
      real Er_old,Fr_old,Er_new,Fr_new;

      real Etot = UcGas(ENG,k,j,i)+UcRad(ER,k,j,i)/reduced_c;
      real mtot = UcGas(MX1,k,j,i)+UcRad(FR1,k,j,i)/reduced_c;
      
      real Er_hyp = UcRad(ER,k,j,i);
      real Fr_hyp = UcRad(FR1,k,j,i);

      real URad[RadiationPhysics::nvar];
      real VRad[RadiationPhysics::nvar];
      real URad_old[RadiationPhysics::nvar];

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
      int count_max = 0;

      while (((err1>tol) || (err2>tol)) && (count < MAX_ITER)){

        for(int nv = 0 ; nv < RadiationPhysics::nvar ; nv++) {
          URad_old[nv] = URad[nv];
          if (std::isnan(URad[nv])){
                   printf("Urad = %e, nv=%i\n",URad[nv],nv);
                   throw std::runtime_error("URad is nan before update");      
          }
        }

        //K_ConsToPrim<DefaultPhysics>(UGas, VGas, &eos);
        //K_ConsToPrim<RadiationPhysics>(URad, VRad, NULL);
        
        //printf("URad[ER] = %e, VRad[ER]=%e, URad[FR1]=%e, VRad[FR1]=%e, VGas[PRS]=%e, UGas[ENG]=%e, VGas[RHO]=%e, UGas[RHO]=%e, \n",URad[ER],VRad[ER],URad[FR1],VRad[FR1],VGas[PRS],UGas[ENG],VGas[RHO],UGas[RHO]);

        real T = VGas[PRS]*unit_energy/(VGas[RHO]*unit_density)/KELVIN;
        //if (count==1) printf("T=%e\n",T);
        //printf("T=%e\n",T);
        real kk_red = reduced_c * unit_velocity * dt * unit_time * kappa_rad * VGas[RHO]*unit_density;
        real xx_red = reduced_c * unit_velocity * dt * unit_time * (xi_rad + kappa_rad) * VGas[RHO]*unit_density;

        URad[ER] = Er_hyp +  kk_red*C_ar*std::pow(T,4)/unit_energy;
        URad[ER] /= 1. + kk_red;
        URad[FR1] = Fr_hyp/(1.+xx_red);

        UGas[ENG] = Etot - URad[ER]/reduced_c;
        UGas[MX1] = mtot - URad[FR1]/reduced_c;

        //real kin = HALF_F / UGas[RHO] * (EXPAND( UGas[MX1]*UGas[MX1]   ,
        //                            + UGas[MX2]*UGas[MX2]  ,
        //                            + UGas[MX3]*UGas[MX3]  ));

        //VGas[PRS] = (gamma-1.0)*(UGas[ENG] - kin);

        //if ( VGas[PRS] < 0.){
        //  printf("WARNING VGas[PRS] = %e\n",VGas[PRS]);
        //  VGas[PRS] = SMALL_PRESSURE_FIX;
          //throw std::runtime_error("VGas[PRS] is negative after ConvToPrim");      
        //}

        //if (i==1) printf("Before ConsToPrim VGas[PRS]=%e, UGas[PRS]=%e\n",VGas[PRS],UGas[PRS]);
        K_ConsToPrim<DefaultPhysics>(VGas, UGas, &eos);
        //if (i==1) printf("After ConsToPrim VGas[PRS]=%e, UGas[PRS]=%e\n",VGas[PRS],UGas[PRS]);

        real Fnorm = std::sqrt(EXPAND(URad[FR1]*URad[FR1] , + URad[FR2]*URad[FR2], + VURad[FR3]*URad[FR3]));
      
        if (Fnorm > URad[ER]) {
        //printf("Limit Flux Fnorm/Er=%e \n",Fnorm/V[ER]);   
        EXPAND( URad[FR1] *= (Fnorm <= 1.e-50 ? URad[ER]/1.e-50 : URad[ER]/Fnorm);, 
                URad[FR2] *= (Fnorm <= 1.e-50 ? URad[ER]/1.e-50 : URad[ER]/Fnorm);,
                URad[FR3] *= (Fnorm <= 1.e-50 ? URad[ER]/1.e-50 : URad[ER]/Fnorm);)
        }

        err1 = std::abs(1.-URad[ER]/URad_old[ER]);
        err2 = std::abs(1.-URad[FR1]/URad_old[FR1]);
        count += 1;
      }
      if (count == MAX_ITER){
        printf("DID NOT CONVERGE: Iterative step took %i cycles at i=%i, URad[ER]=%e and URad[FR1]=%e, err1=%e, err2=%e\n",count,i,URad[ER],URad[FR1],err1,err2);
      }
      
      // to count number of cycles necessary for fixed-point method
      //this->count_max = (this->count_max < count) ? count : this->count_max;

      // if (count == 1){
      //   printf("Did one cycle at i=%i, URad[ER]=%e and URad[FR1]=%e, err1=%e, err2=%e\n",i,URad[ER],URad[FR1],err1,err2);
      // }

      //K_ConsToPrim<DefaultPhysics>(UGas, VGas, &eos);
      //K_ConsToPrim<RadiationPhysics>(URad, VRad, NULL);

      for(int nv = 0 ; nv < RadiationPhysics::nvar ; nv++) {
        UcRad(nv,k,j,i) = URad[nv];
        VcRad(nv,k,j,i) = URad[nv];
      }

      for(int nv = 0 ; nv < DefaultPhysics::nvar ; nv++) {
        UcGas(nv,k,j,i) = UGas[nv];
        VcGas(nv,k,j,i) = VGas[nv];
      }
    });
    //printf("count_max=%i\n",count_max);

  idfx::popRegion();
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
