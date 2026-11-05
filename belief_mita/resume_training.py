"""Exact DataLoader-position and checkpoint checks for interrupted training."""

import hashlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from .schema import FEATURE_DIMS, validate_checkpoint_schema


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def optimizer_steps(optimizer_state):
    steps = [int(state['step']) for state in optimizer_state['state'].values() if 'step' in state]
    if not steps or len(set(steps)) != 1:
        raise ValueError('Adam checkpoint has missing or inconsistent parameter steps')
    return steps[0]


def validate_resume_checkpoint(checkpoint, paths, *, seed, epochs, d, batch_size, observation_version):
    version = validate_checkpoint_schema(checkpoint)
    if version != observation_version or checkpoint.get('d') != d:
        raise ValueError('resume model schema or width differs from the requested training')
    completed = checkpoint.get('epochs')
    in_progress = checkpoint.get('epoch_in_progress')
    batches = checkpoint.get('batches_completed_in_epoch')
    if (type(completed) is not int or type(in_progress) is not int or type(batches) is not int
            or checkpoint.get('status') == 'completed' or not 0 <= completed < epochs
            or in_progress not in (0, completed + 1) or batches < 0
            or (in_progress == 0 and batches != 0)):
        raise ValueError('resume checkpoint has inconsistent epoch progress')
    provenance = checkpoint.get('provenance', {})
    settings = {'seed': seed, 'requested_epochs': epochs, 'batch_size': batch_size,
                'lr': 3e-4, 'observation_version': observation_version,
                'feature_dim': FEATURE_DIMS[observation_version]}
    if any(provenance.get(key) != value for key, value in settings.items()):
        raise ValueError('resume checkpoint training settings differ')
    expected_files = [{'path': str(Path(path).resolve()), 'sha256': sha256(path)} for path in paths]
    if provenance.get('files') != expected_files:
        raise ValueError('resume dataset paths or hashes differ from the original training')
    return {'epochs_completed': completed, 'epoch_in_progress': in_progress,
            'batches_completed_in_epoch': batches, 'provenance': provenance}


def restore_train_iterator(loader, *, completed_epochs, batches_completed_in_epoch):
    """Advance the explicit shuffle generator, then skip only already-updated batches."""
    if loader.num_workers != 0 or loader.batch_size is None or loader.generator is None:
        raise ValueError('exact recovery requires the original single-process generated loader')
    dummy_data = TensorDataset(torch.arange(len(loader.dataset)))
    dummy = DataLoader(dummy_data, batch_size=loader.batch_size, shuffle=True,
                       generator=loader.generator, num_workers=0, drop_last=loader.drop_last)
    for _ in range(completed_epochs):
        for _batch in dummy:
            pass
    iterator = iter(loader)
    if batches_completed_in_epoch > len(loader):
        raise ValueError('checkpoint batch progress exceeds a complete training epoch')
    for _ in range(batches_completed_in_epoch):
        try:
            next(iterator)
        except StopIteration as error:
            raise ValueError('checkpoint batch progress exceeds the shuffled epoch') from error
    return iterator


def resume_train(data_dir, out_dir, checkpoint_path, *, epochs=20, d=64, batch_size=64,
                 seed=20261105, deadline=None, observation_version=1,
                 protocol_sha256=None):
    """Resume from the exact stored optimizer/model update into a fresh output directory."""
    from .training import train
    return train(data_dir, out_dir, epochs=epochs, d=d, batch_size=batch_size, seed=seed,
                 deadline=deadline, observation_version=observation_version,
                 resume_from=checkpoint_path, protocol_sha256=protocol_sha256)
