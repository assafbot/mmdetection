import os

import boto3
from mmengine.runner import CheckpointLoader
from mmengine.runner.checkpoint import load_from_local
from tqdm import tqdm


def _download(local_filename, bucket_name, object_key):
    s3 = boto3.client('s3')
    meta_data = s3.head_object(Bucket=bucket_name, Key=object_key)
    total_length = int(meta_data.get('ContentLength', 0))

    prog_bar = tqdm(total=total_length,
                    unit='B',
                    unit_scale=True, unit_divisor=1024,
                    desc=f'Downloading from s3://{bucket_name}/{object_key}')
    os.makedirs(os.path.dirname(local_filename), exist_ok=True)
    with open(local_filename, 'wb') as f:
        s3.download_fileobj(bucket_name, object_key, f, Callback=prog_bar.update)


@CheckpointLoader.register_scheme(prefixes=('mentee://', ))
def load_from_mentee(filename, map_location=None):
    bucket_name = 'mentee-vision'
    object_key = filename.replace('mentee://', '')
    local_filename = filename.replace('mentee://', os.path.expanduser('~/.mentee/cache/'))

    if not os.path.isfile(local_filename):
        _download(local_filename, bucket_name, object_key)

    return load_from_local(local_filename, map_location=map_location)
