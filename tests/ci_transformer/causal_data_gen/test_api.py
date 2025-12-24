import numpy as np

from ci_transformer.causal_data_gen import (
    CausalSCM,
    SCMMetadata,
    RandomDAGConfig,
    Dataset,
    generate_dataset,
)


def test_generate_dataset_basic_shapes():
    """Smoke test that generate_dataset returns a Dataset and SCM with expected shapes."""
    cfg = RandomDAGConfig(
        d=5,
        edge_prob=0.3,
        max_parents=2,
    )
    ds, scm = generate_dataset(n=100, cfg=cfg, seed=0)

    assert isinstance(ds, Dataset)
    assert isinstance(scm, CausalSCM)
    assert isinstance(ds.meta, SCMMetadata)

    # Dataset has 100 rows and d columns.
    assert ds.X.shape == (100, cfg.d)
    # Metadata is consistent with config.
    assert ds.meta.d == cfg.d
    assert ds.meta.adj.shape == (cfg.d, cfg.d)


def test_generate_dataset_reproducible_with_seed():
    """generate_dataset should be reproducible for fixed cfg and seed."""
    cfg = RandomDAGConfig(
        d=3,
        edge_prob=0.5,
        max_parents=2,
    )
    ds1, scm1 = generate_dataset(n=50, cfg=cfg, seed=123)
    ds2, scm2 = generate_dataset(n=50, cfg=cfg, seed=123)

    # With a fixed config/seed, we expect identical samples and graph.
    assert np.allclose(ds1.X, ds2.X)
    assert np.allclose(scm1.adj, scm2.adj)
    assert ds1.meta.d == ds2.meta.d == cfg.d

