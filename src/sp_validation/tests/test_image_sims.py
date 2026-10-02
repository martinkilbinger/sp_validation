"""UNIT TESTS FOR THE IMAGE-SIMULATION m/c ESTIMATOR.

Exercise ``sp_validation.image_sims.ImageSimMBias`` -- the multiplicative and
additive shear-bias estimator used by the image-simulation workflow -- and the
``sp_validation.catalog.match_catalogs_radec`` helper it relies on.

The estimator recovers ``m`` and ``c`` from five calibrated catalogues named
``1z2z`` (reference, no input shear), ``1p2z``/``1m2z`` (input shear
``g1 = +-|g|``) and ``1z2p``/``1z2m`` (``g2 = +-|g|``). The +g and -g sims are
matched to *each other* by RA/Dec -- same galaxies, opposite input shear -- and
the bias is the object-paired ("pool") average

    m = <(e_+ - e_-) / (2 |g|) - 1> ,   c = <(e_+ + e_-) / 2> ,

so the intrinsic shape cancels object-by-object in ``m``.

We build synthetic catalogues in which the measured ellipticity is exactly
``e = (1 + m_true) g_in + c_true`` at shared positions, so the recovered m/c
must equal the injected values to machine precision -- an analytic check of
the estimator maths that needs no pipeline run.

:Author: cdaley

"""

import numpy as np
import numpy.testing as npt
from astropy.io import fits

from sp_validation.catalog import match_catalogs_radec
from sp_validation.image_sims import ImageSimMBias

# Injected truth, shared across the synthetic-recovery test.
A = 0.02  # input shear amplitude |g|
M_TRUE = 0.05  # multiplicative bias (same for both components)
C1_TRUE = 0.001  # additive bias, component 1
C2_TRUE = -0.002  # additive bias, component 2
N_GAL = 2000


def _write_cat(path, ra, dec, e1, e2, w):
    """Write a minimal calibrated shape catalogue (RA, Dec, e1, e2, w_des)."""
    cols = [
        fits.Column(name=name, array=arr, format="D")
        for name, arr in (
            ("RA", ra),
            ("Dec", dec),
            ("e1", e1),
            ("e2", e2),
            ("w_des", w),
        )
    ]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(
        path, overwrite=True
    )


def _make_grid(grids_dir, num):
    """Create the five sheared/reference catalogues with a known m/c."""
    rng = np.random.default_rng(0)
    ra = 30.0 + rng.uniform(0, 0.1, N_GAL)
    dec = rng.uniform(0, 0.1, N_GAL)
    w = np.ones(N_GAL)
    zero = np.zeros(N_GAL)

    def e(g_in):
        return (1 + M_TRUE) * g_in + zero

    sims = {
        "1z2z": (C1_TRUE + zero, C2_TRUE + zero),
        "1p2z": (e(+A) + C1_TRUE, C2_TRUE + zero),
        "1m2z": (e(-A) + C1_TRUE, C2_TRUE + zero),
        "1z2p": (C1_TRUE + zero, e(+A) + C2_TRUE),
        "1z2m": (C1_TRUE + zero, e(-A) + C2_TRUE),
    }
    for name, (e1, e2) in sims.items():
        sim_dir = grids_dir / f"{name}_grid_{num}"
        sim_dir.mkdir(parents=True, exist_ok=True)
        _write_cat(sim_dir / "cat.fits", ra, dec, e1, e2, w)


def test_match_catalogs_radec_identity():
    """Identical positions match one-to-one; a shifted object drops out."""
    ra = np.array([30.0, 30.01, 30.02])
    dec = np.array([10.0, 10.01, 10.02])
    # Second catalogue = first, but the last object nudged well past threshold.
    ra2, dec2 = ra.copy(), dec.copy()
    ra2[2] += 1.0
    idx1, idx2 = match_catalogs_radec(ra, dec, ra2, dec2, thresh_deg=0.0002)
    npt.assert_array_equal(idx2, [0, 1])
    npt.assert_array_equal(idx1, [0, 1])


def test_mbias_recovers_injected_values(tmp_path):
    """ImageSimMBias recovers the injected m/c to machine precision."""
    num = 7
    _make_grid(tmp_path, num)
    config = {
        "grids_dir": str(tmp_path),
        "num": num,
        "catalog_name": "cat.fits",
        "shear_amplitude": A,
        "match_radius_deg": 0.0002,
        "w_cols": ["w_des"],
        "n_bootstrap": 50,
        "pair_match": True,
        "bootstrap_seed": 42,
    }
    mb = ImageSimMBias(config)
    mb.load_catalogs(verbose=False)
    res = mb.run(verbose=False)

    npt.assert_allclose(res["m1"], M_TRUE, atol=1e-9)
    npt.assert_allclose(res["m2"], M_TRUE, atol=1e-9)
    npt.assert_allclose(res["c1"], C1_TRUE, atol=1e-9)
    npt.assert_allclose(res["c2"], C2_TRUE, atol=1e-9)
    # Bootstrap errors are non-negative and finite.
    for key in ("m1_err", "m2_err", "c1_err", "c2_err"):
        assert np.isfinite(res[key]) and res[key] >= 0
    # Self-describing results: the primary scheme is mirrored at the top level
    # *and* lives under ``weights[scheme]``, and the two agree exactly.
    assert list(res["weights"]) == ["w_des"]
    for key in ("m1", "m1_err", "c1", "c1_err", "m2", "m2_err", "c2", "c2_err"):
        assert res[key] == res["weights"]["w_des"][key]


def test_mbias_multiple_weight_schemes_share_draws(tmp_path):
    """Multi-scheme runs key results per scheme; the primary mirrors the first.

    ``none`` (unit weights) and a real weight column are computed in one run.
    With uniform per-object weights in the synthetic grid the two schemes give
    the *same* m/c (the weighting is a no-op), and the shared bootstrap indices
    make even the errors identical -- the property that lets a scheme
    comparison be a clean weighting comparison. The first entry (``none``) is
    the primary result surfaced at the top level.
    """
    num = 8
    _make_grid(tmp_path, num)
    config = {
        "grids_dir": str(tmp_path),
        "num": num,
        "catalog_name": "cat.fits",
        "shear_amplitude": A,
        "match_radius_deg": 0.0002,
        "w_cols": ["none", "w_des"],
        "n_bootstrap": 50,
        "pair_match": True,
        "bootstrap_seed": 42,
    }
    mb = ImageSimMBias(config)
    mb.load_catalogs(verbose=False)
    res = mb.run(verbose=False)

    assert list(res["weights"]) == ["none", "w_des"]
    # Primary (first) scheme mirrored at the top level.
    for key in ("m1", "m1_err", "c1", "c1_err", "m2", "m2_err", "c2", "c2_err"):
        assert res[key] == res["weights"]["none"][key]
    # Uniform grid weights make the schemes agree bit-for-bit, errors included
    # (shared bootstrap draws).
    assert res["weights"]["none"] == res["weights"]["w_des"]


def test_mbias_deprecated_w_col_still_runs(tmp_path):
    """A pre-``w_cols`` config with the scalar ``w_col`` still runs.

    The deprecated single-scheme key is honoured as ``[w_col]`` when ``w_cols``
    is absent, so a legacy run config keeps working and produces the same
    single-scheme result as the ``w_cols=[w_col]`` spelling.
    """
    num = 9
    _make_grid(tmp_path, num)
    base = {
        "grids_dir": str(tmp_path),
        "num": num,
        "catalog_name": "cat.fits",
        "shear_amplitude": A,
        "match_radius_deg": 0.0002,
        "n_bootstrap": 50,
        "pair_match": True,
        "bootstrap_seed": 42,
    }
    res_dep = ImageSimMBias({**base, "w_col": "w_des"})
    res_dep.load_catalogs(verbose=False)
    out_dep = res_dep.run(verbose=False)

    res_new = ImageSimMBias({**base, "w_cols": ["w_des"]})
    res_new.load_catalogs(verbose=False)
    out_new = res_new.run(verbose=False)

    assert list(out_dep["weights"]) == ["w_des"]
    assert out_dep["weights"] == out_new["weights"]


def test_mbias_pool_cancels_shape_noise(tmp_path):
    """The paired estimator cancels intrinsic shape noise in m.

    With realistic per-galaxy intrinsic ellipticity (sigma_e ~ 0.3) shared
    between the +g and -g sims plus small independent measurement noise, the
    object-paired difference cancels the intrinsic shape, so sigma(m) is set by
    the measurement noise (~1e-2), not the shape noise. An *unpaired* estimator
    (differencing two independently-drawn means) would instead return
    sigma(m) ~ sigma_e / (2 |g| sqrt(N)) -- an order of magnitude larger. We
    assert the recovered error sits well below that shape-noise floor, which is
    the property the pooling exists to deliver.
    """
    num = 3
    rng = np.random.default_rng(1)
    ra = 30.0 + rng.uniform(0, 0.1, N_GAL)
    dec = rng.uniform(0, 0.1, N_GAL)
    w = np.ones(N_GAL)
    sigma_e, sigma_meas = 0.3, 0.01
    e1_int = rng.normal(0, sigma_e, N_GAL)  # intrinsic shape, shared across sims
    e2_int = rng.normal(0, sigma_e, N_GAL)

    def measured(g1_in, g2_in):
        """Measured ellipticity = intrinsic + (1 + m) * input shear + noise."""
        e1 = e1_int + (1 + M_TRUE) * g1_in + rng.normal(0, sigma_meas, N_GAL)
        e2 = e2_int + (1 + M_TRUE) * g2_in + rng.normal(0, sigma_meas, N_GAL)
        return e1, e2

    sims = {
        "1z2z": measured(0, 0),
        "1p2z": measured(+A, 0),
        "1m2z": measured(-A, 0),
        "1z2p": measured(0, +A),
        "1z2m": measured(0, -A),
    }
    for name, (e1, e2) in sims.items():
        sim_dir = tmp_path / f"{name}_grid_{num}"
        sim_dir.mkdir(parents=True, exist_ok=True)
        _write_cat(sim_dir / "cat.fits", ra, dec, e1, e2, w)

    config = {
        "grids_dir": str(tmp_path),
        "num": num,
        "catalog_name": "cat.fits",
        "shear_amplitude": A,
        "match_radius_deg": 0.0002,
        "w_cols": ["w_des"],
        "n_bootstrap": 200,
        "pair_match": True,
        "bootstrap_seed": 42,
    }
    mb = ImageSimMBias(config)
    mb.load_catalogs(verbose=False)
    res = mb.run(verbose=False)

    shape_noise_floor = sigma_e / (2 * A * np.sqrt(N_GAL))  # the unpaired error
    for comp in (1, 2):
        # m recovered within a few sigma of truth...
        assert abs(res[f"m{comp}"] - M_TRUE) < 5 * res[f"m{comp}_err"]
        # ...and its error is far below what an unpaired estimator would give.
        assert res[f"m{comp}_err"] < 0.1 * shape_noise_floor


def _write_jk_cat(path, ra, dec, g_uncal, tiles, R, R_jk, RJ=None, RJ_jk=None):
    """Calibrated catalogue with TILE_ID and a leave-one-tile-out R_JK HDU;
    optionally with the joint response (header RJ_ij, columns RJ_ij)."""
    e = np.linalg.inv(R) @ g_uncal
    cols = [
        fits.Column(name=name, array=arr, format="D")
        for name, arr in (
            ("RA", ra),
            ("Dec", dec),
            ("e1", e[0]),
            ("e2", e[1]),
            ("e1_uncal", g_uncal[0]),
            ("e2_uncal", g_uncal[1]),
        )
    ]
    cols.append(fits.Column(name="TILE_ID", array=tiles, format="7A"))
    primary = fits.PrimaryHDU()
    for i in (1, 2):
        for j in (1, 2):
            primary.header[f"R_{i}{j}"] = R[i - 1, j - 1]
            if RJ is not None:
                primary.header[f"RJ_{i}{j}"] = RJ[i - 1, j - 1]
    labels = sorted(R_jk)
    jk_cols = [fits.Column(name="TILE_ID", array=labels, format="7A")]
    for key, resp in (("R", R_jk), ("RJ", RJ_jk)):
        if resp is None:
            continue
        for i in (1, 2):
            for j in (1, 2):
                jk_cols.append(
                    fits.Column(
                        name=f"{key}_{i}{j}",
                        array=[resp[t][i - 1, j - 1] for t in labels],
                        format="D",
                    )
                )
    fits.HDUList(
        [
            primary,
            fits.BinTableHDU.from_columns(cols),
            fits.BinTableHDU.from_columns(jk_cols, name="R_JK"),
        ]
    ).writeto(path, overwrite=True)


def test_mbias_tile_jackknife_recalibrates_per_tile(tmp_path):
    """The tile jackknife drops one tile from both catalogues, recalibrates
    each with its leave-out response, and uses the (N-1)/N jackknife factor;
    checked against a brute-force computation."""
    num = 3
    n_tiles = 6
    rng = np.random.default_rng(5)
    ra = 30.0 + rng.uniform(0, 0.1, N_GAL)
    dec = rng.uniform(0, 0.1, N_GAL)
    tiles = np.array([f"{100 + i}.200" for i in rng.integers(0, n_tiles, N_GAL)])
    labels = np.unique(tiles)
    e_int = rng.normal(0, 0.3, (2, N_GAL))

    R_true = np.array([[0.75, 0.01], [-0.02, 0.72]])
    g_in = {
        "1z2z": (0, 0),
        "1p2z": (A, 0),
        "1m2z": (-A, 0),
        "1z2p": (0, A),
        "1z2m": (0, -A),
    }
    truth = {}
    for idx, (name, g) in enumerate(g_in.items()):
        g_uncal = R_true @ (e_int + np.array(g)[:, None] * (1 + M_TRUE))
        g_uncal += rng.normal(0, 0.01, (2, N_GAL))
        R = R_true + rng.normal(0, 0.005, (2, 2))
        R_jk = {t: R + rng.normal(0, 0.01, (2, 2)) for t in labels}
        sim_dir = tmp_path / f"{name}_grid_{num}"
        sim_dir.mkdir()
        _write_jk_cat(sim_dir / "cat.fits", ra, dec, g_uncal, tiles, R, R_jk)
        truth[name] = (g_uncal, R_jk)

    config = {
        "grids_dir": str(tmp_path),
        "num": num,
        "catalog_name": "cat.fits",
        "shear_amplitude": A,
        "match_radius_deg": 0.0002,
        "w_cols": ["none"],
        "n_bootstrap": 20,
        "pair_match": True,
        "bootstrap_seed": 42,
    }
    mb = ImageSimMBias(config)
    mb.load_catalogs(verbose=False)
    res = mb.run(verbose=False)

    for name_p, name_m, comp in (("1p2z", "1m2z", 0), ("1z2p", "1z2m", 1)):
        m_jk = []
        for t in labels:
            keep = tiles != t
            (gp, Rp), (gm, Rm) = truth[name_p], truth[name_m]
            ep = np.linalg.solve(Rp[t], gp)[comp][keep]
            em = np.linalg.solve(Rm[t], gm)[comp][keep]
            m_jk.append(np.mean((ep - em) / (2 * A)) - 1)
        m_jk = np.array(m_jk)
        err = np.sqrt((n_tiles - 1) / n_tiles * np.sum((m_jk - m_jk.mean()) ** 2))
        k = comp + 1
        assert res[f"n{k}_jk"] == n_tiles
        npt.assert_allclose(res[f"m{k}_err_jk"], err, rtol=1e-10)
        # TEETH: recalibrating per tile matters; a fixed-R tile jackknife
        # gives a different error.
        (gp, _), (gm, _) = truth[name_p], truth[name_m]
        ep_fix = np.linalg.solve(mb.cats[name_p]["jk"]["resp"]["default"]["R"], gp)[comp]
        em_fix = np.linalg.solve(mb.cats[name_m]["jk"]["resp"]["default"]["R"], gm)[comp]
        m_fix = np.array(
            [np.mean((ep_fix - em_fix)[tiles != t] / (2 * A)) - 1 for t in labels]
        )
        err_fix = np.sqrt((n_tiles - 1) / n_tiles * np.sum((m_fix - m_fix.mean()) ** 2))
        assert not np.isclose(res[f"m{k}_err_jk"], err_fix, rtol=1e-3)


def test_mbias_no_tile_jackknife_without_inputs(tmp_path):
    """Catalogues without TILE_ID / R_JK give bootstrap errors only."""
    num = 8
    _make_grid(tmp_path, num)
    config = {
        "grids_dir": str(tmp_path),
        "num": num,
        "catalog_name": "cat.fits",
        "shear_amplitude": A,
        "match_radius_deg": 0.0002,
        "w_cols": ["w_des"],
        "n_bootstrap": 10,
        "pair_match": True,
        "bootstrap_seed": 42,
    }
    mb = ImageSimMBias(config)
    mb.load_catalogs(verbose=False)
    res = mb.run(verbose=False)
    assert not any(key.endswith("_jk") for key in res)


def test_mbias_joint_response(tmp_path):
    """With the joint response on input, m/c are computed a second time with
    the shear recalibrated by R_joint, including the tile jackknife; the
    default results are unchanged."""
    num = 4
    n_tiles = 5
    rng = np.random.default_rng(9)
    ra = 30.0 + rng.uniform(0, 0.1, N_GAL)
    dec = rng.uniform(0, 0.1, N_GAL)
    tiles = np.array([f"{100 + i}.200" for i in rng.integers(0, n_tiles, N_GAL)])
    labels = np.unique(tiles)
    e_int = rng.normal(0, 0.3, (2, N_GAL))

    R_true = np.array([[0.75, 0.01], [-0.02, 0.72]])
    g_in = {
        "1z2z": (0, 0),
        "1p2z": (A, 0),
        "1m2z": (-A, 0),
        "1z2p": (0, A),
        "1z2m": (0, -A),
    }
    truth = {}
    for name, g in g_in.items():
        g_uncal = R_true @ (e_int + np.array(g)[:, None] * (1 + M_TRUE))
        R = R_true + rng.normal(0, 0.005, (2, 2))
        RJ = R + rng.normal(0, 0.005, (2, 2))
        R_jk = {t: R + rng.normal(0, 0.01, (2, 2)) for t in labels}
        RJ_jk = {t: RJ + rng.normal(0, 0.01, (2, 2)) for t in labels}
        sim_dir = tmp_path / f"{name}_grid_{num}"
        sim_dir.mkdir()
        _write_jk_cat(
            sim_dir / "cat.fits", ra, dec, g_uncal, tiles, R, R_jk, RJ, RJ_jk
        )
        truth[name] = (g_uncal, R, RJ, RJ_jk)

    config = {
        "grids_dir": str(tmp_path),
        "num": num,
        "catalog_name": "cat.fits",
        "shear_amplitude": A,
        "match_radius_deg": 0.0002,
        "w_cols": ["none"],
        "n_bootstrap": 20,
        "pair_match": True,
        "bootstrap_seed": 42,
    }
    mb = ImageSimMBias(config)
    mb.load_catalogs(verbose=False)
    res = mb.run(verbose=False)
    joint = res["joint"]["weights"]["none"]

    for name_p, name_m, comp in (("1p2z", "1m2z", 0), ("1z2p", "1z2m", 1)):
        (gp, Rp, RJp, RJp_jk), (gm, Rm, RJm, RJm_jk) = truth[name_p], truth[name_m]
        k = comp + 1

        # Central values: default with R, joint with R_joint
        for resp_p, resp_m, out in ((Rp, Rm, res), (RJp, RJm, joint)):
            ep = np.linalg.solve(resp_p, gp)[comp]
            em = np.linalg.solve(resp_m, gm)[comp]
            npt.assert_allclose(out[f"m{k}"], np.mean((ep - em) / (2 * A)) - 1)
            npt.assert_allclose(out[f"c{k}"], np.mean((ep + em) / 2), atol=1e-15)

        # Joint jackknife uses the joint leave-out responses
        m_jk = np.array(
            [
                np.mean(
                    (
                        np.linalg.solve(RJp_jk[t], gp)[comp]
                        - np.linalg.solve(RJm_jk[t], gm)[comp]
                    )[tiles != t]
                    / (2 * A)
                )
                - 1
                for t in labels
            ]
        )
        err = np.sqrt((n_tiles - 1) / n_tiles * np.sum((m_jk - m_jk.mean()) ** 2))
        npt.assert_allclose(joint[f"m{k}_err_jk"], err, rtol=1e-10)
        assert joint[f"m{k}"] != res[f"m{k}"]
