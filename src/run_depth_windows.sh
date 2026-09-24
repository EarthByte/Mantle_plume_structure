#!/bin/bash
for zr in "2000 2700" "2300 2700"; do
  set -- $zr
  echo "########## $1 to $2 km ##########"
  python3 conduit_detect.py --file ../../REVEAL_mantle_tomography/RevealLO.nc \
      --tag RevealLO --var voigt --every 2 --depth-max 2880 --nulls 300 \
      --zmin "$1" --zmax "$2" --suffix "_z$1"
done
