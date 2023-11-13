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
  
  IdefixArray4D<real> Vrad;      // Main cell-centered radiation variables

  DataBlock *data;
  
 //private:
  


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

  //if constexpr(Phys::radiation) {
  Vrad = IdefixArray4D<real>(prefix+"_Vrad", 1+DIMENSIONS,
              this->data->np_tot[KDIR]+KOFFSET, this->data->np_tot[JDIR]+JOFFSET, this->data->np_tot[IDIR]+IOFFSET);
  //}


};




#endif //RAD_HPP_