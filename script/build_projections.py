"""
Builds the PLS projection matrices required by activation_perturbing.py /
activation_steering.py from the activations extracted by
activation_extraction.py.

Replicates the `load_emb` / `visualze_activation` logic of
notebook/jupyter_script_for_visualization.ipynb (without the plotting):
activations of the three input formats (table / temp / story) are
concatenated, a 10-component PLS regression is fit against the ordinal
indices (i) and relational indices (r), and the QR-orthonormalized
x-weights are saved as {llm_tp}_l{layer}_{i,r}_{data_tp}_proj.npy.
"""
import argparse
import os

import numpy as np
import torch
from sklearn.cross_decomposition import PLSRegression


def load_emb(sd, data_tp, input_tp, layer=15, nb=200, mod="llama"):
    sf = os.path.join(sd, f"{mod}_l{layer}_{data_tp}_{input_tp}.pt")
    data = torch.load(sf, map_location="cpu")

    xs_atts = [data[f"att{i}_{j}"].cpu().numpy()[:nb, :]
               for i in range(1, 5) for j in range(1, 4)]
    embs = np.concatenate(xs_atts, axis=0)

    yi = data["yi"].cpu().numpy()[:nb]
    xi = data["xi"].cpu().numpy()[:nb]
    zi = data["zi"].cpu().numpy()[:nb]

    yis, xis, zis = [], [], []
    for i in range(1, 5):
        for j in range(1, 4):
            # atts4 rows take their y-index from xi (as in the notebook)
            yis.extend((xi if i == 4 else yi)[:, j - 1])
            xis.extend(xi[:, i])
            # atts1 rows take their z-index from xi (as in the notebook)
            zis.extend((xi if i == 1 else zi)[:, i])

    inds_yx = np.stack((yis, xis), axis=1)
    inds_yz = np.stack((yis, zis), axis=1)
    return embs, inds_yx, inds_yz


def build_projections(sd, data_tp, layer=15, mod="llama", n_components=10):
    embs_ta, inds_ta_i, inds_ta_r = load_emb(sd, data_tp, "table", layer=layer, mod=mod)
    embs_te, inds_te_i, inds_te_r = load_emb(sd, data_tp, "temp", layer=layer, mod=mod)
    embs_st, inds_st_i, inds_st_r = load_emb(sd, data_tp, "story", layer=layer, mod=mod)

    embs = np.concatenate([embs_ta, embs_te, embs_st], axis=0)
    inds_i = np.concatenate([inds_ta_i, inds_te_i, inds_st_i], axis=0)
    inds_r = np.concatenate([inds_ta_r, inds_te_r, inds_st_r], axis=0)

    for tp, inds in (("i", inds_i), ("r", inds_r)):
        pls_model = PLSRegression(n_components=n_components)
        pls_model.fit(embs, inds)
        V_re, _ = np.linalg.qr(pls_model.x_weights_)
        sf = os.path.join(sd, f"{mod}_l{layer}_{tp}_{data_tp}_proj.npy")
        np.save(sf, V_re)
        print(f"saved {sf} shape {V_re.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build PLS projection matrices")
    parser.add_argument("--llm_tp", type=str, default="llama", help="llama / qwen / mistral")
    parser.add_argument("--layer", type=int, default=15, help="")
    parser.add_argument("--sd_emb", type=str, default="./data_emb/", help="")
    args = parser.parse_args()

    for data_tp in ["space", "create", "job", "relation", "city"]:
        build_projections(args.sd_emb, data_tp, layer=args.layer, mod=args.llm_tp)
