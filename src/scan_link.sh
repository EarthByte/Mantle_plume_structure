#!/bin/bash
for L in 250 400 600; do
  echo "########## link ${L} km ##########"
  python3 deflect_structures.py --file ../../REVEAL_mantle_tomography/RevealLO.nc \
      --tag RevealLO --var voigt --every 2 --depth-max 2880 --mode floor \
      --threshold 0.65 --floor-sites 8 --link-km $L --suffix "_L$L" 2>&1 \
    | sed -n '/FLOOR:/,/^out\//p'
done
