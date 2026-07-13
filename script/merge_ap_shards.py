import argparse
import glob
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge ap_result_*_shard{i}of{n}.json files produced by "
                    "parallel runs of activation_sampling.py")
    parser.add_argument("--sd_result", type=str, default="./result/sample/")
    args = parser.parse_args()

    shard_files = glob.glob(os.path.join(args.sd_result, "ap_result_*_shard*of*.json"))
    groups = {}
    for sf in shard_files:
        base = sf[:sf.rindex("_shard")] + ".json"
        groups.setdefault(base, []).append(sf)

    for base, shards in sorted(groups.items()):
        merged = {}
        for sf in sorted(shards):
            with open(sf) as f:
                merged.update(json.load(f))
        with open(base, "w") as f:
            json.dump(merged, f)
        print(f"{base}: merged {len(shards)} shards, {len(merged)} grids")
