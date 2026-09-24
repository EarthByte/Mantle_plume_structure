#!/bin/bash
T=../../REVEAL_mantle_tomography
run () {
  tag=$1; file=$2; var=$3; every=$4; zmax=$5
  echo "===== $tag ====="
  python3 morph_run.py --file "$file" --tag "$tag" --var "$var" \
      --every "$every" --depth-max "$zmax" --targets morph_targets.txt \
      --corridor --configs 8 --box-deg 20 --suffix _xm --jobs 2 || return
  python3 component_continuity.py --tag "$tag" --suffix _xm
}
run SEMUCB-WM1 "$T/SEMUCB-WM1.nc" vs    1 2891
run SPiRaL     "$T/SPiRaL.nc"     voigt 1 2891
run GLADM35    "$T/GLADM35.nc"    voigt 2 2890
