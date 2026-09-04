from pathlib import Path


def assert_disjoint_patch_pools(training: list[Path], unseen: list[Path]) -> None:
    """Reject a benchmark when training and unseen pools share patch IDs."""
    train_ids = {path.stem for path in training}
    unseen_ids = {path.stem for path in unseen}
    overlap = train_ids & unseen_ids
    if overlap:
        raise ValueError(f"training/unseen patch overlap: {sorted(overlap)}")
