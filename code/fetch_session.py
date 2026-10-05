"""Download one Dynamic Routing session from the public AIND bucket, laid out like the
Code Ocean datacube mount so the notebooks only need their `root` path changed.

    python fetch_session.py 743199_2024-12-05 ~/data/dynamicrouting_datacube

Needs no AWS credentials -- s3://aind-open-data is public.
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "aind-open-data"
N_THREADS = 16


def _client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED, max_pool_connections=N_THREADS))


def find_asset(s3, session_id):
    """Full asset prefix (e.g. 'ecephys_743199_..._nwb_2026-08-04_15-00-49/') for '<subject>_<date>'."""
    r = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"ecephys_{session_id}", Delimiter="/")
    prefixes = [p["Prefix"] for p in r.get("CommonPrefixes", []) if "_nwb_" in p["Prefix"]]
    if not prefixes:
        raise SystemExit(f"no NWB asset found for {session_id}")
    return sorted(prefixes)[-1]  # most recent build


def download(session_id, dest_root):
    s3 = _client()
    asset = find_asset(s3, session_id)
    dest_root = Path(dest_root).expanduser()
    print(f"s3://{BUCKET}/{asset} -> {dest_root / asset.rstrip('/')}")

    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=asset):
        keys += [(o["Key"], o["Size"]) for o in page.get("Contents", [])]
    total = sum(size for _, size in keys)
    print(f"{len(keys)} objects, {total / 1e9:.2f} GB")

    done = [0]

    def get(item):
        key, size = item
        out = dest_root / key
        if out.exists() and out.stat().st_size == size:
            done[0] += 1
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET, key, str(out))
        done[0] += 1
        if done[0] % 200 == 0:
            print(f"  {done[0]}/{len(keys)}", flush=True)

    with ThreadPoolExecutor(N_THREADS) as pool:
        list(pool.map(get, keys))
    print(f"done: {done[0]}/{len(keys)} objects")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    download(sys.argv[1], sys.argv[2])
