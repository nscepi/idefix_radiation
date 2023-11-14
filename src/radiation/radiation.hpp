// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef RAD_HPP_
#define RAD_HPP_

#include <string>
#include <vector>
#include <memory>

#include "idefix.hpp"
#include "grid.hpp"
#include "dataBlock.hpp"
#include "physics.hpp"

template<typename Phys>
class Radiation {
 public:
  
  // Data related to current instance of the Fluid object
  std::string prefix;
  int instanceNumber;

  Radiation( Grid &, Input&, DataBlock *, int n = 0);
  void ConvertConsToPrim();
  void ConvertPrimToCons();
  void EvolveStage(const real, const real);
  void ResetStage();

  IdefixArray4D<real> Vrad;      // Main cell-centered radiation primitive variables
  IdefixArray4D<real> Urad;      // Main cell-centered radiation conservative variables

// Required by time integrator
  IdefixArray3D<real> InvDt;

  DataBlock *data;
  
 private:
  
  // Loop on dimensions
  template <int dir>
  void LoopDir(const real, const real);


};


template<typename Phys>
Radiation<Phys>::Radiation(Grid &grid, Input &input, DataBlock *datain, int n) {

  idfx::pushRegion("Radiation::Radiation");
  // Save the datablock to which we are attached from now on
  this->data = datain;

  // Create our own prefix
  prefix = std::string(Phys::prefix);

  // Keep the instance # for later use
  instanceNumber = n;


  // We now allocate the fields required by the radiation solver
  Vrad = IdefixArray4D<real>(prefix+"_Vrad", 1+DIMENSIONS,
              this->data->np_tot[KDIR]+KOFFSET, this->data->np_tot[JDIR]+JOFFSET, this->data->np_tot[IDIR]+IOFFSET);
  Urad = IdefixArray4D<real>(prefix+"_Urad", 1+DIMENSIONS,
              this->data->np_tot[KDIR]+KOFFSET, this->data->np_tot[JDIR]+JOFFSET, this->data->np_tot[IDIR]+IOFFSET);
  InvDt = IdefixArray3D<real>(prefix+"_InvDt",
                              this->data->np_tot[KDIR], this->data->np_tot[JDIR], this->data->np_tot[IDIR]);
 
  
  

};


#include "evolveStage.hpp"
#include "enroll.hpp"
#include "convertConsToPrim.hpp"


#endif //RAD_HPP_