#!/bin/bash
T=../../REVEAL_mantle_tomography
SCAN="0.40 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.95 1.10"
run () {
  tag=$1; file=$2; var=$3; every=$4; zmax=$5
  echo "############ $tag ############"
  python3 prominence_map.py --file "$file" --tag "$tag" --var "$var" \
      --every "$every" --depth-max "$zmax" --nulls 300 --scan $SCAN
  echo "---- per-depth prominence excess, $tag ----"
  python3 conduit_detect.py --file "$file" --tag "$tag" --var "$var" \
      --every "$every" --depth-max "$zmax" --nulls 300 --suffix "_xm" \
      2>&1 | tail -3
}
run SEMUCB-WM1 "$T/SEMUCB-WM1.nc" vs    1 2891
run SPiRaL     "$T/SPiRaL.nc"     voigt 1 2891
run GLADM35    "$T/GLADM35.nc"    voigt 2 2890
