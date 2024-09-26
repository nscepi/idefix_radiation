#!/usr/bin/env python3

"""

@author: glesur
"""
import os
import sys
sys.path.append(os.getenv("IDEFIX_DIR"))

import pytools.idfx_test as tst

name="dump.0001.dmp"

tolerance=1e-10

def testMe(test):

  deffiles = ["definitions-cart3D.hpp","definitions-sph1D.hpp"]
  inifiles=["idefix-hll-sph.ini","idefix-hll-cart.ini"]

  # loop on all the ini files for this test
  for definition in deffiles:
    if ("cart" in definition):
      inifiles=["idefix-hll-cart.ini"]
    elif ("sph" in definition):
      inifiles=["idefix-hll-sph.ini"]
    for ini in inifiles:
      test.configure(definitionFile=definition)
      test.compile()
      test.run(inputFile=ini)
      if test.init and not test.mpi:
        test.makeReference(filename=name)
      test.standardTest()
      test.nonRegressionTest(filename=name,tolerance=tolerance)


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
