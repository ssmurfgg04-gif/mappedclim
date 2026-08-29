#!/bin/bash

# Download tract polygons and road layers for all 4 regions from Source Cooperative proxy
set -e
BASE="https://data.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge"
D="${MAPPEDCLIM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}/data"
cd $D

dl() {  # dl <region> <layer>
  local region=$1 layer=$2
  local src="$BASE/reference/$region/$region-$layer.parquet"
  local dst="$D/roads/${region}-${layer}.parquet"
  if [ "$layer" = "census-tracts" ]; then dst="$D/tracts/$region-census-tracts.parquet"; fi
  if [[ "$layer" == *buildings* ]]; then dst="$D/buildings/${region}-${layer}.parquet"; fi
  if [ -f "$dst" ]; then echo "SKIP $dst"; return; fi
  echo "Downloading $region-$layer ..."
  curl -sL --retry 3 --retry-delay 5 "$src" -o "$dst" -w "  done: %{size_download} bytes (%{speed_download} B/s)\n"
}

# Tracts (small)
for region in eastern-ok maricopa-az northern-ca south-central-tx; do
  dl $region census-tracts &
done
wait

# Roads (parallel per region: overture-roads + tiger-roads)
for region in eastern-ok maricopa-az northern-ca south-central-tx; do
  dl $region overture-roads &
  dl $region census-tiger-roads &
done
wait

# Buildings (~3.7 GB total: Overture + Microsoft footprints per region)
for region in eastern-ok maricopa-az northern-ca south-central-tx; do
  dl $region overture-buildings &
done
wait
for region in eastern-ok maricopa-az northern-ca south-central-tx; do
  dl $region microsoft-buildings &
done
wait
echo "=== ALL DOWNLOADS COMPLETE ==="
ls -la $D/tracts/ $D/roads/ $D/buildings/
