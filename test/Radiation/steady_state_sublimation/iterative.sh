#!/bin/bash

cp 'dump.hr002.dmp' 'dump.ini.dmp'

for i in {0..15}
do
  mpirun --mca io ^ompio -np 2 ./idefix -dec 2 1
  cp 'dump.0001.dmp' 'dump.ini.dmp'
  cp 'dump.0001.dmp' 'dump.cycle'$i'.dmp'
  mkdir 'cycle'$i
  mv data.*.vtk 'cycle'$i/
done




