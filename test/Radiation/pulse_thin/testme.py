#!/usr/bin/env python3

"""

@author: glesur
"""
import os
import sys
import shutil

sys.path.append(os.getenv("IDEFIX_DIR"))

import pytools.idfx_test as tst

name="dump.0001.dmp"

tolerance=1e-10

def testMe(test):

  deffiles = ["definitions-sph1D.hpp","definitions-sph2D.hpp","definitions-sph3D.hpp","definitions-cart3D.hpp"]

  # loop on all the ini files for this test
  for definition in deffiles:
    if ("cart" in definition):
      if ("1D" in definition):
        shutil.copy('idefix-hll-cart.ini', 'idefix-hll-cart1D.ini')
        shutil.copy('idefix-hllc-cart.ini', 'idefix-hllc-cart1D.ini')
        inifiles=["idefix-hll-cart1D.ini","idefix-hllc-cart1D.ini"]
      elif ("2D" in definition):
        shutil.copy('idefix-hll-cart.ini', 'idefix-hll-cart2D.ini')
        shutil.copy('idefix-hllc-cart.ini', 'idefix-hllc-cart2D.ini')
        inifiles=["idefix-hll-cart2D.ini","idefix-hllc-cart2D.ini"]
      elif ("3D" in definition):
        shutil.copy('idefix-hll-cart.ini', 'idefix-hll-cart3D.ini')
        shutil.copy('idefix-hllc-cart.ini', 'idefix-hllc-cart3D.ini')
        inifiles=["idefix-hll-cart3D.ini","idefix-hllc-cart3D.ini"]
    elif ("sph" in definition):
      if ("1D" in definition):
        shutil.copy('idefix-hll-sph.ini', 'idefix-hll-sph1D.ini')
        shutil.copy('idefix-hllc-sph.ini', 'idefix-hllc-sph1D.ini')
        inifiles=["idefix-hll-sph1D.ini","idefix-hllc-sph1D.ini"]
      elif ("2D" in definition):
        shutil.copy('idefix-hll-sph.ini', 'idefix-hll-sph2D.ini')
        shutil.copy('idefix-hllc-sph.ini', 'idefix-hllc-sph2D.ini')
        inifiles=["idefix-hll-sph2D.ini","idefix-hllc-sph2D.ini"]
      elif ("3D" in definition):
        shutil.copy('idefix-hll-sph.ini', 'idefix-hll-sph3D.ini')
        shutil.copy('idefix-hllc-sph.ini', 'idefix-hllc-sph3D.ini')
        inifiles=["idefix-hll-sph3D.ini","idefix-hllc-sph3D.ini"]

    test.configure(definitionFile=definition)
    test.compile()
    for ini in inifiles:
        test.run(inputFile=ini)
        if test.init and not test.mpi:
            test.makeReference(filename=name)
        test.standardTest()
        test.nonRegressionTest(filename=name,tolerance=tolerance)
        os.remove(ini)


test=tst.idfxTest()
if not test.dec:
  test.dec=['2']

if not test.all:
  if(test.check):
    test.checkOnly(filename=name,tolerance=tolerance)
  else:
    testMe(test)
else:
  for rec in range(1,4):
    test.vectPot=False
    test.single=False
    test.reconstruction=rec
    test.mpi=False
    testMe(test)

  test.mpi=True
  testMe(test)
