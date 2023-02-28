from mmengine.runner.checkpoint import load_from_local

try:
    import clearml
except ImportError:
    clearml = None

from mmengine.runner import CheckpointLoader


@CheckpointLoader.register_scheme(prefixes=('mentee://', ))
def load_from_mentee(filename, map_location=None):
    if clearml is None:
        raise ImportError('Please run "pip install clearml" to install clearml')

    remote_filename = filename.replace('mentee://', 's3://mentee-vision/')
    local_filename = clearml.StorageManager.get_local_copy(remote_filename)
    return load_from_local(local_filename, map_location=map_location)
