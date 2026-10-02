"""IMAGE_SIMS.

:Description: Multiplicative and additive shear bias from image simulations.

:Author: Martin Kilbinger

"""

import numpy as np
from astropy.io import fits

from sp_validation.catalog import match_catalogs_radec

# Conventional campaign layout, used only when the config carries no branch map
# (e.g. the synthetic-recovery tests).  In a workflow run the branches and pairs
# come from manifest.yaml via the m_bias config; nothing about the injected
# shear is hard-coded on the estimator's side.
_DEFAULT_BRANCHES = ["1z2z", "1p2z", "1m2z", "1z2p", "1z2m"]
_DEFAULT_PAIRS = [
    ("1p2z", "1m2z", 0),  # g1 component, index 0 → e1
    ("1z2p", "1z2m", 1),  # g2 component, index 1 → e2
]


# Weight-scheme name that means "no weighting": every object gets unit weight.
# ``None`` (from a YAML ``null``) is accepted as an alias, so the fiducial
# unweighted primary scheme can be written either ``none`` or ``null``.
_UNWEIGHTED = "none"


def _is_unweighted(scheme):
    """True for the unit-weight scheme (``"none"`` or ``None``)."""
    return scheme is None or scheme == _UNWEIGHTED


def _load_cat(path, w_cols):
    """Load RA, Dec, ellipticities and per-scheme weights from a FITS catalogue.

    Reads the ``e1``/``e2`` columns, which the calibration stage writes as the
    *calibrated* shear estimate ``g = R^-1 g_uncal - c`` (metacal response and
    additive-bias corrected) -- not the raw ``e1_uncal``/``e2_uncal`` columns
    that sit alongside them in the same catalogue.  The bias this estimator
    measures is therefore the *residual* m/c left after the chain's own metacal
    calibration, not the raw pre-calibration bias.

    ``w_cols`` is the list of weight schemes to load. The scheme ``"none"``
    (equivalently a ``None``/``null`` entry) gives every object unit weight --
    the no-weighting mode for m-bias runs (#227: shape weights are excluded from
    sim calibration); any other entry is read as a FITS column name. The weights
    come back as a dict keyed by scheme so one catalogue load serves every
    scheme in a multi-weight run.

    If the header has the joint response ``RJ_ij`` (written by the
    calibration stage), the shear recalibrated with it is loaded as
    ``e1_joint``/``e2_joint``. If the catalogue has a ``TILE_ID`` column and an
    ``R_JK`` HDU (the leave-one-tile-out responses), the inputs for a
    jackknife over tiles are loaded under ``cat["jk"]``; see
    ``_load_jackknife``.
    """
    with fits.open(path) as hdul:
        data = hdul[1].data
        cat = {
            "ra": data["RA"].copy(),
            "dec": data["Dec"].copy(),
            "e1": data["e1"].copy(),
            "e2": data["e2"].copy(),
            "w": {},
        }
        for scheme in w_cols:
            cat["w"][scheme] = (
                np.ones(len(cat["ra"]))
                if _is_unweighted(scheme)
                else data[scheme].copy()
            )
        has_joint = "RJ_11" in hdul[0].header
        has_jk = "TILE_ID" in data.columns.names and "R_JK" in hdul
        if has_joint or has_jk:
            R, g_off = _uncalibrated(hdul, cat, path)
        if has_joint:
            R_joint = _matrix(hdul[0].header, "RJ")
            cat["e1_joint"], cat["e2_joint"] = np.linalg.inv(R_joint) @ g_off
        if has_jk:
            cat["jk"] = _load_jackknife(hdul, R, R_joint if has_joint else None, g_off)
        return cat


def _matrix(row, key):
    """2x2 matrix from the entries ``<key>_11`` ... ``<key>_22`` of ``row``."""
    return np.array(
        [[row[f"{key}_11"], row[f"{key}_12"]], [row[f"{key}_21"], row[f"{key}_22"]]]
    )


def _uncalibrated(hdul, cat, path):
    """Response and uncalibrated shear, to recalibrate a catalogue.

    The calibrated shear is ``e = R^-1 g_uncal - s``, where the shift ``s`` is
    the additive correction (zero when ``additive_correction`` is off, as for
    constant-shear sims). Returns R and ``g_off = g_uncal - R s``, so that a
    response R' recalibrates as ``e' = R'^-1 g_off``, which gives back ``e``
    for R' = R.
    """
    data = hdul[1].data
    R = _matrix(hdul[0].header, "R")
    g_uncal = np.array([data["e1_uncal"], data["e2_uncal"]], dtype=float)

    shift = np.linalg.inv(R) @ g_uncal - np.array([cat["e1"], cat["e2"]])
    if np.any(np.ptp(shift, axis=1) > 1e-8):
        raise ValueError(
            f"{path}: e1/e2 are not R^-1 e_uncal minus a constant, cannot"
            + " recalibrate"
        )
    return R, g_uncal - (R @ shift.mean(axis=1))[:, None]


def _load_jackknife(hdul, R, R_joint, g_off):
    """Inputs to recalibrate a catalogue with a leave-one-tile-out response.

    A jackknife sample without tile t is recalibrated as
    ``e_t = R_t^-1 g_off``, see ``_uncalibrated``; the same for the joint
    response, if available (``R_joint`` not ``None`` and the ``RJ_ij``
    columns present).
    """
    data = hdul[1].data
    jk = hdul["R_JK"].data
    tiles = [str(t).strip() for t in jk["TILE_ID"]]

    resp = {"default": (R, "R")}
    if R_joint is not None and "RJ_11" in jk.columns.names:
        resp["joint"] = (R_joint, "RJ")
    return {
        "tile": np.char.strip(np.asarray(data["TILE_ID"]).astype(str)),
        "g_off": g_off,
        "resp": {
            method: {
                "R": R_full,
                "R_jk": {t: _matrix(jk[i], key) for i, t in enumerate(tiles)},
            }
            for method, (R_full, key) in resp.items()
        },
    }


def _jackknife_std(x):
    """Leave-one-out jackknife standard deviation."""
    n = len(x)
    return np.sqrt((n - 1) / n * np.sum((x - np.mean(x)) ** 2))


class ImageSimMBias:
    """Compute multiplicative and additive shear bias from image simulations.

    The estimator consumes the *calibrated* ``e1``/``e2`` columns (the metacal
    response- and additive-bias-corrected shear ``g = R^-1 g_uncal - c``), so
    the headline m/c is the **residual** bias remaining after the chain's own
    metacal calibration, not the raw pre-calibration bias.

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys:
        - grids_dir : str, path to the grids directory
        - num : int, run number (e.g. 2 for *_grid_2)
        - catalog_name : str, filename of the cut catalogue
          (default 'shape_catalog_cut_ngmix.fits')
        - shear_amplitude : float, input shear |g| (from manifest.yaml)
        - branches : list of str, branch names in load order (incl. the
          unsheared reference); defaults to the conventional 5-branch layout
        - pairs : list of dicts {plus, minus, component}, the +/- sheared
          branch pairing per component; defaults to the conventional pairs
        - match_radius_deg : float, matching radius in degrees (required)
        - pair_match : bool, match objects between the +g and -g sheared
          catalogues (required); if False, use all objects of each
          catalogue (the paired per-object cancellation is then unavailable)
        - w_cols : list of str, weight schemes to compute in one run
          (required); ``"none"`` (or a ``null`` entry) means unit weights,
          any other entry is a FITS column name. The **first** entry is the
          primary result surfaced at the top level of ``run()``'s output. Our
          fiducial run leads with the unweighted scheme (``["none", ...]``),
          per the #227 verdict that shape weights are excluded from sim
          calibration; the unweighted m also avoids the ``cov(w, e)`` residual
          weighted estimators carry on constant-shear sims.
        - w_col : str or None, *deprecated* single weight scheme; accepted for
          back-compat and used as ``[w_col]`` only when ``w_cols`` is absent.
        - n_bootstrap : int, number of bootstrap resamples for errors (required)
        - bootstrap_seed : int, seed for the per-pair bootstrap RNG (required);
          makes the bootstrap errors bit-reproducible. The resample indices are
          drawn once per pair and shared across every weight scheme, so the
          schemes differ only in their weighting, never in their draws.

    The science knobs (``match_radius_deg``, ``pair_match``, ``w_cols``,
    ``n_bootstrap``, ``bootstrap_seed``) are read with no in-code default: a
    missing one is a config bug and raises ``KeyError`` at construction, per the
    fail-fast contract (the workflow emits every one into the m_bias config).
    The lone exception is the deprecated ``w_col``, which is honoured as a
    fallback so pre-``w_cols`` configs still run.
    """

    def __init__(self, config):
        self.cfg = config
        self.g_in = config["shear_amplitude"]
        self.thresh = config["match_radius_deg"]
        self.pair_match = config["pair_match"]
        # ``w_cols`` is the required science key. A pre-``w_cols`` config that
        # still carries the deprecated scalar ``w_col`` is honoured as a
        # single-scheme run; only a config with neither raises (fail-fast).
        if "w_cols" in config:
            w_cols = config["w_cols"]
        else:
            w_cols = [config["w_col"]]
        # Normalise a ``None``/``null`` entry to the canonical "none" name so
        # results key off a string; downstream still treats it as unit weights.
        self.w_cols = [_UNWEIGHTED if _is_unweighted(w) else str(w) for w in w_cols]
        self.n_boot = config["n_bootstrap"]
        self.boot_seed = config["bootstrap_seed"]
        # Branch list and pairing come from the manifest-derived config
        # (``branches`` / ``pairs``); fall back to the conventional layout only
        # when neither is given.  ``branches`` fixes the catalogue load order;
        # ``pairs`` fixes which sims difference into which component.
        self.sim_names = list(config.get("branches", _DEFAULT_BRANCHES))
        if config.get("pairs"):
            self.pairs = [
                (p["plus"], p["minus"], p["component"]) for p in config["pairs"]
            ]
        else:
            self.pairs = list(_DEFAULT_PAIRS)
        self.cats = {}

    def load_catalogs(self, verbose=True):
        """Load the 5 sheared and reference catalogues."""
        grids_dir = self.cfg["grids_dir"]
        num = self.cfg["num"]
        cat_name = self.cfg.get("catalog_name", "shape_catalog_cut_ngmix.fits")
        # Grid sims live in ``{branch}_grid_{num}``; other families (e.g.
        # blended) in ``{branch}_{num}``.
        sims_type = self.cfg.get("sims_type", "grid")
        suffix = f"_grid_{num}" if sims_type == "grid" else f"_{num}"
        # ``sim_names`` (incl. the unsheared reference) comes from the config's
        # branch map. The +g/-g pool estimator pairs the sheared sims directly;
        # the reference is loaded for completeness and null-test diagnostics.
        for name in self.sim_names:
            path = f"{grids_dir}/{name}{suffix}/{cat_name}"
            if verbose:
                print(f"  Loading {path}")
            self.cats[name] = _load_cat(path, self.w_cols)
            if verbose:
                print(f"    {len(self.cats[name]['ra'])} objects")

    def print_mean_ellipticities(self):
        """Print the mean e1, e2 for each catalogue and weight scheme, as a check.

        The unweighted scheme (``"none"``) gives the plain unweighted means.
        """
        for scheme in self.w_cols:
            print(f"\nMean ellipticities (all objects, weights: {scheme}):")
            for name, cat in self.cats.items():
                mean_e1 = np.average(cat["e1"], weights=cat["w"][scheme])
                mean_e2 = np.average(cat["e2"], weights=cat["w"][scheme])
                print(f"  {name}:  <e1> = {mean_e1:+.5f}   <e2> = {mean_e2:+.5f}")

    def _m_c_pair(self, name_p, name_m, comp, verbose=True, method="default"):
        """Compute m and c for one shear pair and component (0=g1, 1=g2).

        Paired ("pool") estimator. The +g and -g simulations inject opposite
        input shear on the *same* galaxies, so matching them directly by
        RA/Dec yields a one-to-one correspondence. Differencing the two
        ellipticities per object,

            m = <(e_+ - e_-) / (2 g_in) - 1> ,   c = <(e_+ + e_-) / 2> ,

        cancels the intrinsic shape (sigma_e ~ 0.3) object-by-object in the
        multiplicative term, leaving only measurement noise -- so sigma(m)
        shrinks by ~sigma_e/sigma_meas relative to differencing two
        independent means. (The additive term c is a *sum*, so intrinsic
        shape does not cancel there and its error stays shape-noise limited.)

        With ``pair_match=False`` the +g and -g sims are *not* matched: every
        object of each catalogue is used, so the per-object cancellation is
        lost and m, c fall back to differencing/summing the two independent
        weighted means. The paired bootstrap likewise cannot be applied (the
        two arrays generally have different lengths), so each side is resampled
        independently per replicate.

        ``method`` selects the response the shear is calibrated with:
        "default" (R = R_shear + R_selection, the catalogue's e1/e2) or
        "joint" (R_joint, see ``metacal._response_from_selections``).
        """
        e_key = f"e{comp + 1}" + ("_joint" if method == "joint" else "")

        if self.pair_match:
            # Match the +g and -g sims to each other: same galaxies, opposite
            # shear. This is a nearest-neighbour match within `thresh`, not a
            # strict bijection -- on grid sims galaxies are well separated so
            # pairs are effectively 1:1 (verified ~99% co-located to <0.05" on
            # SKiLLS grid_1); on denser fields a small fraction could share a
            # +g partner and dilute the cancellation.
            idx_p, idx_m = match_catalogs_radec(
                self.cats[name_p]["ra"],
                self.cats[name_p]["dec"],
                self.cats[name_m]["ra"],
                self.cats[name_m]["dec"],
                thresh_deg=self.thresh,
            )
            if verbose:
                print(f"  {name_p} <-> {name_m}: {len(idx_p)} paired objects")
        else:
            idx_p = slice(None)
            idx_m = slice(None)
            if verbose:
                print(
                    f"  no pair-matching: {name_p}: {len(self.cats[name_p][e_key])}"
                    f"  |  {name_m}: {len(self.cats[name_m][e_key])} objects"
                )

        e_p = self.cats[name_p][e_key][idx_p]
        e_m = self.cats[name_m][e_key][idx_m]

        # Draw the bootstrap resample indices *once*, before the weight-scheme
        # loop, and reuse them for every scheme -- the schemes then differ only
        # in their weighting, never in their draws (so a scheme comparison is a
        # clean weighting comparison). Pre-drawing the full ``(n_boot, n)`` block
        # in one call is bit-identical to drawing ``rng.integers(0, n, n)`` once
        # per replicate (numpy fills the block row-major), so the numbers match a
        # single-scheme, per-iteration bootstrap to the last bit.
        rng = np.random.default_rng(seed=self.boot_seed)
        if self.pair_match:
            n = len(e_p)
            ib = rng.integers(0, n, (self.n_boot, n))
        else:
            n_p, n_m = len(e_p), len(e_m)
            ib_p = rng.integers(0, n_p, (self.n_boot, n_p))
            ib_m = rng.integers(0, n_m, (self.n_boot, n_m))

        res = {}
        for scheme in self.w_cols:
            w_p = self.cats[name_p]["w"][scheme][idx_p]
            w_m = self.cats[name_m]["w"][scheme][idx_m]
            m_boot = np.empty(self.n_boot)
            c_boot = np.empty(self.n_boot)

            if self.pair_match:
                # Per-object shear-differenced (-> m) and summed (-> c)
                # ellipticity, with a symmetric per-pair weight.
                w = 0.5 * (w_p + w_m)
                d = (e_p - e_m) / (2 * self.g_in) - 1
                s = (e_p + e_m) / 2

                m = np.average(d, weights=w)
                c = np.average(s, weights=w)

                # Paired bootstrap: the same object draw is applied to both
                # sims, so the per-object cancellation in `d` is preserved in
                # the error estimate.
                for i in range(self.n_boot):
                    m_boot[i] = np.average(d[ib[i]], weights=w[ib[i]])
                    c_boot[i] = np.average(s[ib[i]], weights=w[ib[i]])
            else:
                # No matching: difference/sum the two independent weighted means.
                mean_ep = np.average(e_p, weights=w_p)
                mean_em = np.average(e_m, weights=w_m)

                m = (mean_ep - mean_em) / (2 * self.g_in) - 1
                c = (mean_ep + mean_em) / 2

                # Unpaired bootstrap: the +g and -g arrays generally differ in
                # length, so each side is resampled independently per replicate.
                for i in range(self.n_boot):
                    ep_b = np.average(e_p[ib_p[i]], weights=w_p[ib_p[i]])
                    em_b = np.average(e_m[ib_m[i]], weights=w_m[ib_m[i]])
                    m_boot[i] = (ep_b - em_b) / (2 * self.g_in) - 1
                    c_boot[i] = (ep_b + em_b) / 2

            res[scheme] = {
                "m": m,
                "m_err": np.std(m_boot),
                "c": c,
                "c_err": np.std(c_boot),
            }

        if all(
            method in self.cats[name].get("jk", {}).get("resp", {})
            for name in (name_p, name_m)
        ):
            self._jackknife_pair(name_p, name_m, comp, idx_p, idx_m, res, method)
        elif verbose:
            print("  no tile IDs or R_JK on input, skipping tile jackknife")

        return res

    def _jackknife_pair(self, name_p, name_m, comp, idx_p, idx_m, res, method):
        """Leave-one-tile-out jackknife errors on m and c, added to ``res``.

        Each sample drops one tile from both the +g and the -g catalogue, and
        recalibrates each catalogue with the response computed without that
        tile. The error thus includes the noise of R_shear and R_selection,
        which the bootstrap over calibrated ellipticities (fixed R) misses.
        Tiles are resampled as a whole, which keeps correlations within a tile.
        ``method`` selects the response, as in ``_m_c_pair``.
        """
        jk_p = self.cats[name_p]["jk"]
        jk_m = self.cats[name_m]["jk"]
        resp_p = jk_p["resp"][method]
        resp_m = jk_m["resp"][method]
        tile_p = jk_p["tile"][idx_p]
        tile_m = jk_m["tile"][idx_m]
        g_off_p = jk_p["g_off"][:, idx_p]
        g_off_m = jk_m["g_off"][:, idx_m]
        tiles = np.union1d(tile_p, tile_m)

        m_jk = {scheme: np.empty(len(tiles)) for scheme in self.w_cols}
        c_jk = {scheme: np.empty(len(tiles)) for scheme in self.w_cols}
        for i, tile in enumerate(tiles):
            # Recalibrated component comp, with the tile's leave-out response
            # (the full R if the tile is absent from that catalogue)
            row_p = np.linalg.inv(resp_p["R_jk"].get(tile, resp_p["R"]))[comp]
            row_m = np.linalg.inv(resp_m["R_jk"].get(tile, resp_m["R"]))[comp]
            e_p = row_p @ g_off_p
            e_m = row_m @ g_off_m

            for scheme in self.w_cols:
                w_p = self.cats[name_p]["w"][scheme][idx_p]
                w_m = self.cats[name_m]["w"][scheme][idx_m]
                if self.pair_match:
                    keep = (tile_p != tile) & (tile_m != tile)
                    w = 0.5 * (w_p + w_m)[keep]
                    d = (e_p[keep] - e_m[keep]) / (2 * self.g_in) - 1
                    s = (e_p[keep] + e_m[keep]) / 2
                    m_jk[scheme][i] = np.average(d, weights=w)
                    c_jk[scheme][i] = np.average(s, weights=w)
                else:
                    keep_p = tile_p != tile
                    keep_m = tile_m != tile
                    mean_ep = np.average(e_p[keep_p], weights=w_p[keep_p])
                    mean_em = np.average(e_m[keep_m], weights=w_m[keep_m])
                    m_jk[scheme][i] = (mean_ep - mean_em) / (2 * self.g_in) - 1
                    c_jk[scheme][i] = (mean_ep + mean_em) / 2

        for scheme in self.w_cols:
            res[scheme]["m_err_jk"] = _jackknife_std(m_jk[scheme])
            res[scheme]["c_err_jk"] = _jackknife_std(c_jk[scheme])
            res[scheme]["n_jk"] = len(tiles)

    def run(self, verbose=True):
        """Compute m and c for both shear components and every weight scheme.

        Returns
        -------
        dict
            ``results["weights"][scheme]`` holds ``m1, m1_err, c1, c1_err,
            m2, m2_err, c2, c2_err`` for each weight scheme. The primary
            (first) scheme's keys are also mirrored at the top level, so a
            reader that wants the headline m/c never has to know the scheme
            name. If the inputs support a tile jackknife, the keys
            ``m1_err_jk, c1_err_jk, n1_jk`` (and for comp 2) are added. If the
            inputs carry the joint response, ``results["joint"]["weights"]``
            holds the same keys with the shear calibrated by R_joint.
        """
        results = {"weights": self._run_method("default", verbose)}

        names = {name for pair in self.pairs for name in pair[:2]}
        if all("e1_joint" in self.cats[name] for name in names):
            results["joint"] = {"weights": self._run_method("joint", verbose)}
        elif verbose:
            print("\nNo joint response on input, skipping method 'joint'")

        # Mirror the primary (first) scheme's m/c at the top level: the headline
        # result reads out without knowing the scheme name, and a downstream
        # gate keyed on the old flat keys still finds them.
        results.update(results["weights"][self.w_cols[0]])
        return results

    def _run_method(self, method, verbose):
        """m and c per weight scheme, for the response ``method``."""
        weights = {scheme: {} for scheme in self.w_cols}
        for name_p, name_m, comp in self.pairs:
            label = f"g{comp + 1}"
            if verbose:
                print(f"\n--- {label}: {name_p} / {name_m}, response {method} ---")
            res = self._m_c_pair(name_p, name_m, comp, verbose=verbose, method=method)
            for scheme, r in res.items():
                w = weights[scheme]
                k = comp + 1
                w[f"m{k}"] = r["m"]
                w[f"m{k}_err"] = r["m_err"]
                w[f"c{k}"] = r["c"]
                w[f"c{k}_err"] = r["c_err"]
                if "n_jk" in r:
                    w[f"m{k}_err_jk"] = r["m_err_jk"]
                    w[f"c{k}_err_jk"] = r["c_err_jk"]
                    w[f"n{k}_jk"] = r["n_jk"]
                if verbose:
                    print(
                        f"  [{scheme}] m{k} = {r['m']:.4f} ± {r['m_err']:.4f}"
                        f"   c{k} = {r['c']:.4f} ± {r['c_err']:.4f}"
                    )
                    if "n_jk" in r:
                        print(
                            f"  [{scheme}] tile jackknife ({r['n_jk']} tiles):"
                            f" m{k} ± {r['m_err_jk']:.4f}"
                            f"   c{k} ± {r['c_err_jk']:.4f}"
                        )
        return weights
