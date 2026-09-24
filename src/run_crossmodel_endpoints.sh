#!/bin/bash
T=../../REVEAL_mantle_tomography
run () {
  tag=$1; file=$2; var=$3; every=$4; zmax=$5
  [ -f "$file" ] || { echo "!! $tag: $file missing"; return; }
  echo "===== $tag ====="
  python3 root_bias.py --mode endpoints --tag "$tag" --file "$file" \
      --var "$var" --every "$every" --depth-max "$zmax" \
      --province-vote out/province_vote.npz --min-votes 3 \
      --nulls 300 --resolution 375
}
run SEMUCB-WM1 "$T/SEMUCB-WM1.nc" vs    1 2891
run SPiRaL     "$T/SPiRaL.nc"     voigt 1 2891
run GLADM35    "$T/GLADM35.nc"    voigt 2 2890
run REVEAL     "$T/REVEAL.nc"     voigt 1 2880
