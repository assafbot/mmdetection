import clearml
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--task-name', help='clearml task name', default=None)
    parser.add_argument('--project-name', help='clearml project name', default='mmdetection')
    parser.add_argument('--queue', help='clearml queue', default='default')
    args, unknownargs = parser.parse_known_args()

    task_name = args.task_name or os.path.splitext(os.path.basename(args.config))[0]
    task = clearml.Task.init(project_name=args.project_name, task_name=task_name, output_uri='s3://mentee-vision/mmdetection/clearml/')
    unknownargs = ' '.join(unknownargs)
    unknownargs = task.connect({'args': unknownargs})['args']

    # TODO: @assaf support connecting config so it can be modified from clearml
    # args.config = task.connect_configuration(args.config)

    task.execute_remotely(queue_name=args.queue)

    tools_dir = os.path.dirname(__file__)
    dist_script = os.path.join(tools_dir, 'dist_train.sh')
    cmd = f'{dist_script} {args.config} ' + unknownargs
    print(f'Executing: {cmd}')
    os.system(cmd)


if __name__ == '__main__':
    main()
