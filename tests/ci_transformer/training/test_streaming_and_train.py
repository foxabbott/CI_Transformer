import torch

from ci_transformer.training import (
    CIStreamingDataset,
    Curriculum,
    make_dataloader,
    train_ci_model,
)


def test_ci_streaming_dataset_iterates():
    """Basic smoke test that CIStreamingDataset yields CI dicts with expected shapes."""
    n_rows = 16
    m_max = 3
    curriculum = Curriculum()  # use default curriculum
    dataset = CIStreamingDataset(
        n_rows=n_rows,
        d_min=3,
        d_max=3,
        m_max=m_max,
        curriculum=curriculum,
        seed=0,
        include_meta=True,
        scm_refresh=1,
        data_refresh=1,
    )

    item = next(iter(dataset))

    # Check minimal structural properties; do not bake in distributional assumptions.
    assert isinstance(item["x"], torch.Tensor)
    assert isinstance(item["y"], torch.Tensor)
    assert isinstance(item["z"], torch.Tensor)
    assert isinstance(item["z_mask"], torch.Tensor)
    assert isinstance(item["label"], torch.Tensor)

    assert item["x"].shape == (n_rows,)
    assert item["y"].shape == (n_rows,)
    assert item["z"].shape == (m_max, n_rows)
    assert item["z_mask"].shape == (m_max,)

    # When include_meta=True, a small debugging dict should be present.
    assert "meta" in item
    meta = item["meta"]
    assert {"i", "j", "S", "d", "m"}.issubset(set(meta.keys()))


def test_make_dataloader_collate_shapes():
    """make_dataloader should collate CIStreamingDataset items into batched tensors."""
    n_rows = 8
    m_max = 2
    batch_size = 4
    dataset = CIStreamingDataset(
        n_rows=n_rows,
        d_min=3,
        d_max=3,
        m_max=m_max,
        seed=1,
        include_meta=False,
        scm_refresh=1,
        data_refresh=1,
    )
    loader = make_dataloader(dataset, batch_size=batch_size, num_workers=0, pin_memory=False)

    batch = next(iter(loader))

    assert batch["x"].shape == (batch_size, n_rows)
    assert batch["y"].shape == (batch_size, n_rows)
    assert batch["z"].shape == (batch_size, m_max, n_rows)
    assert batch["z_mask"].shape == (batch_size, m_max)
    assert batch["label"].shape == (batch_size,)


def test_train_ci_model_smoke(tmp_path):
    """Smoke test that train_ci_model can run a tiny CPU-only training loop and write outputs."""
    out_dir = tmp_path / "ci_run"
    train_ci_model(
        out_dir=str(out_dir),
        steps=2,
        batch_size=4,
        n_rows=32,
        m_max=2,
        num_workers=0,
        device="cpu",
        seed=42,
    )

    # Check that key artifacts were written.
    assert (out_dir / "config.json").is_file()
    assert (out_dir / "final.pt").is_file()

