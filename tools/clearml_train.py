import argparse
import os
import shutil

import clearml


def main():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--task-name', help='clearml task name', default=None)
    parser.add_argument('--project-name', help='clearml project name', default='mmdetection')
    parser.add_argument('--queue', help='clearml queue', default='default')
    args, unknownargs = parser.parse_known_args()

    assert os.path.isfile(args.config), f'{args.config} does not exist'

    task_name = args.task_name or os.path.splitext(os.path.basename(args.config))[0]
    task = clearml.Task.init(project_name=args.project_name, task_name=task_name, output_uri='s3://mentee-vision/mmdetection/clearml/')
    unknownargs = ' '.join(unknownargs)
    unknownargs = task.connect({'args': unknownargs})['args']

    task.execute_remotely(queue_name=args.queue)

    if task is not None:
        if 'Config' in task.get_configuration_objects().keys():
            tmp_path = task.connect_configuration('', name='Config')
            config_name = os.path.basename(args.config)
            config_path = os.path.join(os.path.dirname(tmp_path), config_name)
            shutil.copyfile(tmp_path, config_path)
            args.config = config_path
            print(f'Using configuration from task {task.name} ({task.task_id})')

    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    port = 29600
    if visible_devices is not None:
        visible_devices_idx = list(map(int, visible_devices.split(',')))
        port += sum(map(lambda x: 2**x, visible_devices_idx))

    print('CUDA_VISIBLE_DEVICES:', visible_devices)

    tools_dir = os.path.dirname(__file__)
    dist_script = os.path.join(tools_dir, 'dist_train.sh')
    cmd = f'PORT={port} {dist_script} {args.config} ' + unknownargs
    print(f'Executing: {cmd}')
    failed = os.system(cmd)
    assert not failed


if __name__ == '__main__':
    main()
