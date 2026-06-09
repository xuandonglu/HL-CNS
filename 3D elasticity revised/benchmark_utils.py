from pathlib import Path


def ensure_parent_dir(path: str) -> None:
    parent = Path(path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)


def resolve_existing_path(primary: str, *fallbacks: str) -> str:
    """Return the first existing path while keeping a new primary path in code."""
    for candidate in (primary, *fallbacks):
        if candidate and Path(candidate).exists():
            if candidate != primary:
                print(f"[Compatibility] Using legacy path: {candidate}")
            return str(candidate)
    return primary


def benchmark_tag_from_path(path: str, default: str = "run") -> str:
    for part in Path(path).parts:
        if part.startswith("model_"):
            return part.removeprefix("model_")
    stem = Path(path).stem
    for prefix in ("best_hlcns_", "ckpt_hlcns_", "checkpoint_hlcns_", "best_", "ckpt_", "checkpoint_"):
        if stem.startswith(prefix):
            return stem.removeprefix(prefix)
    return default


def print_checkpoint_load(path: str, is_checkpoint_dict: bool) -> None:
    kind = "checkpoint dict" if is_checkpoint_dict else "state_dict"
    print(f"[Loaded {kind}] {path}")


def print_param_count(trainable_params: int, total_params: int) -> None:
    print(f"Parameters -- trainable: {trainable_params:,} | total: {total_params:,}")


def print_metrics(split: str, mse: float, rel: float) -> None:
    print(f"{split:<5} -- MSE: {mse:.6e} | Rel: {rel:.6e}")


def print_time(split: str, seconds: float) -> None:
    print(f"{split:<5} -- Time: {seconds:.4f} s")


def export_mat_batch(
    enabled: bool,
    output_dir: str,
    model_name: str,
    tag: str,
    split: str,
    batch_index: int,
    tensors: dict,
):
    """Export the first batch of a split for external plotting when requested."""
    if not enabled or batch_index != 0:
        return None

    from scipy.io import savemat

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_{tag}_{split.lower()}.mat"
    payload = {}
    for key, value in tensors.items():
        payload[key] = value.detach().cpu().numpy() if hasattr(value, "detach") else value
    savemat(str(out_path), payload)
    print(f"[Exported MAT] {out_path}")
    return str(out_path)
