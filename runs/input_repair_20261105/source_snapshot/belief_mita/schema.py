"""Versioned observation input dimensions for checkpoints and episode archives."""

FEATURE_DIMS = {0: 18, 1: 37}


def validate_checkpoint_schema(checkpoint):
    """Return checkpoint observation version, rejecting silent dimension mismatches."""
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a mapping")
    version = checkpoint.get("observation_version", 0)
    feature_dim = checkpoint.get("feature_dim", 18)
    if type(version) is not int or type(feature_dim) is not int:
        raise ValueError("checkpoint observation schema must use integer version and dimension")
    if version not in FEATURE_DIMS or FEATURE_DIMS[version] != feature_dim:
        raise ValueError(f"unsupported observation schema: version={version}, feature_dim={feature_dim}")
    return version
