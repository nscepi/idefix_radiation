// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************
#ifndef FLUID_RADSOURCE_HPP_
#define FLUID_RADSOURCE_HPP_

#include <string>
#include "idefix.hpp"
#include "input.hpp"
#include "fluid_defs.hpp"
#include "eos.hpp"
#include "units.hpp"
#include "lookupTable.hpp"
#include "column.hpp"

class RadSource {
 public:
  enum class Type_opac{constant,kramers,usertable};
  enum class Type_irr{constant,usertable};
  enum class Type_isolver{full_implicit,fixed_point_rad,fixed_point_gas};
  // Different types of implementation for the radiation source terms.
  template <typename Phys>
  RadSource(Input &, Fluid<Phys> *);
  void ShowConfig();                    // print configuration
  void AddRadSource(const real);
  void Source_full_implicit(const real);
  void Source_fixed_point_rad(const real);
  void Source_fixed_point_gas(const real);
  void IrrFlux(const real);

  IdefixArray4D<real> UcRad;  // Radiation conservative quantities
  IdefixArray4D<real> UcGas;  // Gas conservative quantities
  IdefixArray4D<real> VcRad;  // Radiation primitive quantities
  IdefixArray4D<real> VcGas;  // Gas primitive quantities
  IdefixArray3D<real> InvDt;  // The InvDt of current radiation multigroup
  
  // Compute kappa_p
  KOKKOS_INLINE_FUNCTION void K_kappa_p(int i, int j, int k, real* kappa_p) const {
    auto units = idfx::units;

    if (this->kappa_type == Type_opac::constant){
      real temp = this->kappa_0;
      *kappa_p = temp;
    } else if (this->kappa_type == Type_opac::kramers){
      real T = this->VcGas(PRS,k,j,i)/(this->VcGas(RHO,k,j,i))*units.Kelvin*this->mu;
      real temp = kappa_0*std::pow(this->VcGas(RHO,k,j,i)*units.density/rho_0,2.)*std::pow(T/T_0,-3.5);
      *kappa_p = temp;
    } else if (this->kappa_type == Type_opac::usertable){
      real T = this->VcGas(PRS,k,j,i)/(this->VcGas(RHO,k,j,i))*units.Kelvin*this->mu;
      real logT = std::log10(T);
      auto k_p = this->kappa_planck_1D;
      real temp = k_p.Get(&logT);
      *kappa_p = temp;
    }
  }
  
  // Compute kappa_r
  KOKKOS_INLINE_FUNCTION void K_kappa_r(int i, int j, int k, real* kappa_r) const {
    auto units = idfx::units;

    if (kappa_type == Type_opac::constant){
      *kappa_r = this->kappa_0;
    } else if (kappa_type == Type_opac::kramers){
      real T = this->VcGas(PRS,k,j,i)/(VcGas(RHO,k,j,i))*units.Kelvin*this->mu;
      *kappa_r = kappa_0*std::pow(this->VcGas(RHO,k,j,i)*units.density/rho_0,2.)*std::pow(T/T_0,-3.5);
    } else if (kappa_type == Type_opac::usertable){
      real T = this->VcGas(PRS,k,j,i)/(this->VcGas(RHO,k,j,i))*units.Kelvin*this->mu;
      real logT = std::log10(T);
      auto k_r = this->kappa_ross_1D;
      *kappa_r = k_r.Get(&logT);
    }
  }

  // Compute xi
  KOKKOS_INLINE_FUNCTION void K_xi(int i, int j, int k, real* xi) const {
    auto units = idfx::units;

    if (xi_type == Type_opac::constant){
      *xi = this->xi_0;
    } else if (xi_type == Type_opac::usertable){
      real T = this->VcGas(PRS,k,j,i)/(this->VcGas(RHO,k,j,i))*units.Kelvin*this->mu;
      real logT = std::log10(T);
      auto xi1D = this->xi_1D;
      *xi = xi1D.Get(&logT);
    }
  }

  // Compute limiting diffusion speed for Riemann solver in opt. thick media
  KOKKOS_INLINE_FUNCTION real Limit_speeds_Rad(int i, int j, int k, real dx) const {
    auto VcGas = this->VcGas;
    real kappa,xi;
    auto units=idfx::units;

    real unit_density = units.density;
    real unit_length = units.length;
    
    K_kappa_p(i,j,k,&kappa);
    K_xi(i,j,k,&xi);

    // Compute optical depth across one cell
    real tau = VcGas(RHO,k,j,i)*units.density*(kappa+xi)*dx*units.length;

    // return characteristic velocity of radiative diffusion 
    return 4./(3.*tau)*this->reduced_c;
  };

   IdefixArray3D<real> GetdivF() {
    return (this->divF);
  }

 private:
  DataBlock* data;
  real kappa_0;
  real xi_0;
  real kappa_irr;
  real rho_0;
  real T_0;
  real rs;
  real Ts;
  real reduced_c;
  real gamma;
  real mu;
  int count_max;
  
  // Sound speed computation
  EquationOfState *eos;

  // Dimension of Planck and Rosseland opacities tables
  int kappa_ndim;
  int xi_ndim;

  // Opacity tables
  LookupTable<1> kappa_planck_1D;
  LookupTable<1> kappa_ross_1D;
  LookupTable<1> xi_1D;

  // Have irradiation or not
  bool haveIrradiation{false};

  // Dimension of irradiation flux table 
  int irr_ndim;
  
  // Irradiation flux table
  LookupTable<1> irr_1D;

  Column *column_rho;
  IdefixArray3D<real> tau;  // column density
  IdefixArray3D<real> divF;  // Divergence of irradiation flux
 

  Type_isolver source_solver;
  Type_opac kappa_type;
  Type_opac xi_type;
  Type_irr irr_type;

};

#include "fluid.hpp"

template<typename Phys>
RadSource::RadSource(Input &input, Fluid<Phys> *hydroin):
                      UcRad{hydroin->Uc},
                      UcGas{hydroin->data->hydro->Uc},
                      VcRad{hydroin->Vc},
                      VcGas{hydroin->data->hydro->Vc},
                      InvDt{hydroin->InvDt} {
  idfx::pushRegion("RadSource::RadSource");
  
  // Save the parent hydro object
  this->data = hydroin->data;
  this->eos = hydroin->data->hydro->eos.get();

  // Check in which block we should fetch our information
  std::string BlockName;
  if(Phys::radiation) {
    BlockName = "Rad";
  } else {
    IDEFIX_ERROR("Fluid is not radiative");
  }

  // Reduced velocity of light 
  this->reduced_c =  hydroin->reduced_c;
  
  // Adiabatic index
  this->gamma = this->eos->GetGamma();
  
  // Mean molecular weight
  this->mu = this->eos->GetMu();

  // Information on scattering opacity coefficient
  if(input.CheckEntry(BlockName,"xi")>=0) {
    // Fetch the opacity coefficient for the current radiation group.
    const int n = hydroin->instanceNumber;

    std::string xiType = input.Get<std::string>(BlockName,"xi",0);
    if(xiType.compare("constant") == 0) {
      this->xi_type = Type_opac::constant;
      this->xi_0 = input.Get<real>(BlockName,"xi",n+1);
    } else if(xiType.compare("usertable") == 0) {
      this->xi_type = Type_opac::usertable;
      this->xi_ndim = input.Get<int>(BlockName,"xi",n+1);
      std::string xi_file = input.Get<std::string>(BlockName,"xi",n+2);
      if (input.Get<int>(BlockName,"xi",n+1) == 1){
        this->xi_1D = LookupTable<1>(xi_file,',');
      } else {
        std::stringstream msg;
        msg << "Only 1 dimension for scattering opacity tables are currently accepted." << std::endl;
        IDEFIX_ERROR(msg);
      }
    } else {
      std::stringstream msg;
      msg << "Unknown xi type \"" <<  xiType
          << "\" in your input file." << std::endl
          << "Allowed values are: constant, usertable." << std::endl;

      IDEFIX_ERROR(msg);
    }
  } else {
    IDEFIX_ERROR("A *xi* line in your [Rad] block is required in your input file to define the scattering opacity.");
  }
  
  // Information on absorption opacity coefficient
  if(input.CheckEntry(BlockName,"kappa")>=0) {
    // Fetch the opacity coefficient for the current radiation group.
    const int n = hydroin->instanceNumber;

    std::string kappaType = input.Get<std::string>(BlockName,"kappa",0);
    if(kappaType.compare("constant") == 0) {
      this->kappa_type = Type_opac::constant;
      this->kappa_0 = input.Get<real>(BlockName,"kappa",n+1);
    } else if(kappaType.compare("kramers") == 0) {
      this->kappa_0 = input.Get<real>(BlockName,"kappa",n+1);
      this->kappa_type = Type_opac::kramers;
      this->rho_0 = input.Get<real>(BlockName,"kappa",n+2);
      this->T_0 = input.Get<real>(BlockName,"kappa",n+3);
    } else if(kappaType.compare("usertable") == 0) {
      this->kappa_type = Type_opac::usertable;
      this->kappa_ndim = input.Get<int>(BlockName,"kappa",n+1);
      std::string kappap_file = input.Get<std::string>(BlockName,"kappa",n+2);
      std::string kappar_file = input.Get<std::string>(BlockName,"kappa",n+3);
      if (input.Get<int>(BlockName,"kappa",n+1) == 1){
        this->kappa_planck_1D = LookupTable<1>(kappap_file,',');
        this->kappa_ross_1D = LookupTable<1>(kappar_file,',');
      } else {
        std::stringstream msg;
        msg << "Only 1 dimension for absorption opacity tables are currently accepted." << std::endl;
        IDEFIX_ERROR(msg);
      }
      
    } else {
      std::stringstream msg;
      msg << "Unknown kappa type \"" <<  kappaType
          << "\" in your input file." << std::endl
          << "Allowed values are: constant, kramers, usertable." << std::endl;

      IDEFIX_ERROR(msg);
    }
  } else {
    IDEFIX_ERROR("A *kappa* line in your [Rad] block is required in your input file to define the absorption opacity.");
  }

  // Information on solver for source terms
  if(input.CheckEntry(BlockName,"source")>=0) {
    // Fetch the opacity coefficient for the current radiation group.
    const int n = hydroin->instanceNumber;

    std::string sourceType = input.Get<std::string>(BlockName,"source",0);
    if(sourceType.compare("full_implicit") == 0) {
      this->source_solver = Type_isolver::full_implicit;
    } else if(sourceType.compare("fixed_point_rad") == 0) {
      this->source_solver = Type_isolver::fixed_point_rad;
    } else if(sourceType.compare("fixed_point_gas") == 0) {
      this->source_solver = Type_isolver::fixed_point_gas;      
    } else {
      std::stringstream msg;
      msg << "Unknown solver for source terms \"" <<  sourceType
          << "\" in your input file." << std::endl
          << "Allowed values are: full_implicit, fixed_point_rad, fixed_point_gas." << std::endl;

      IDEFIX_ERROR(msg);
    }

  } else {
    IDEFIX_ERROR("A *source* line in your [Rad] block is required in your input file to define the solver for the radiation source terms.");
  }

  // Information on irradiation source term
  if(input.CheckEntry(BlockName,"irr")>=0) {
    haveIrradiation = true;
    // Fetch the opacity coefficient for the current radiation group.
    const int n = hydroin->instanceNumber;
    
    this->column_rho = new Column(IDIR,1,RHO,data);
    this->divF = IdefixArray3D<real>("divF",data->np_tot[KDIR], data->np_tot[JDIR], data->np_tot[IDIR]);

    std::string irrType = input.Get<std::string>(BlockName,"irr",0);
    
    if(irrType.compare("constant") == 0) {
      this->irr_type = Type_irr::constant;
      this->rs = input.Get<real>(BlockName,"irr",n+1);
      this->Ts = input.Get<real>(BlockName,"irr",n+2);
      this->kappa_irr = input.Get<real>(BlockName,"irr",n+3);
    } else if(irrType.compare("usertable") == 0) {
      this->irr_type = Type_irr::usertable;
      this->rs = input.Get<real>(BlockName,"irr",n+1);
      this->Ts = input.Get<real>(BlockName,"irr",n+2);
      this->irr_ndim = input.Get<int>(BlockName,"irr",n+3);
      std::string irr_file = input.Get<std::string>(BlockName,"irr",n+4);
      if (input.Get<int>(BlockName,"irr",n+3) == 1){
        this->irr_1D = LookupTable<1>(irr_file,',');
      } else {
        std::stringstream msg;
        msg << "Only 1 dimension for irradiation flux tables are currently accepted." << std::endl;
        IDEFIX_ERROR(msg);
      }
    } else {
      std::stringstream msg;
      msg << "Unknown irr type \"" <<  irrType
          << "\" in your input file." << std::endl
          << "Allowed values are: constant, usertable." << std::endl;

      IDEFIX_ERROR(msg);
    }
  }
  


  idfx::popRegion();
}

#endif // FLUID_RADSOURCE_HPP_
