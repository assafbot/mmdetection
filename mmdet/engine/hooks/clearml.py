import os
from typing import Optional, Dict

from mmengine.dist import master_only
from mmengine.hooks import Hook, CheckpointHook

from mmdet.registry import HOOKS


@HOOKS.register_module()
class ClearMLLoggerHook(Hook):
    def __init__(self, init_kwargs: Optional[Dict] = None, task_type=None) -> None:
        self.clearml = None
        self.task = None
        self.init_kwargs = init_kwargs
        self.import_clearml()
        self.task_type = self.clearml.TaskTypes(task_type or 'training')

    def import_clearml(self):
        try:
            import clearml
        except ImportError:
            raise ImportError('Please run "pip install clearml" to install clearml')
        self.clearml = clearml

    @master_only
    def before_run(self, runner) -> None:
        task_kwargs = self.init_kwargs if self.init_kwargs else {}
        if 'task_name' not in task_kwargs:
            task_kwargs['task_name'] = runner.experiment_name
        assert 'task_type' not in task_kwargs
        task_kwargs['task_type'] = self.task_type

        # Use current task or create a new one
        self.task = self.clearml.Task.current_task() or self.clearml.Task.init(**task_kwargs)

        assert runner.cfg.filename is not None and runner.work_dir is not None
        cfg_dump_path = os.path.join(runner.work_dir, os.path.basename(runner.cfg.filename))
        assert os.path.isfile(cfg_dump_path)
        self.task.connect_configuration(cfg_dump_path, name='Config')


@HOOKS.register_module()
class ClearMLCheckpointHook(CheckpointHook):
    def __init__(self, init_kwargs: Optional[Dict] = None, task_type=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.clearml = None
        self.task = None
        self.import_clearml()

    def import_clearml(self):
        try:
            import clearml
        except ImportError:
            raise ImportError('Please run "pip install clearml" to install clearml')
        self.clearml = clearml

    def _save_checkpoint(self, runner) -> None:
        """Save the current checkpoint and delete outdated checkpoint.

        Args:
            runner (Runner): The runner of the training process.
        """
        super()._save_checkpoint(runner)
        self._upload(runner)

    @master_only
    def _upload(self, runner):
        if self.by_epoch:
            ckpt_filename = self.filename_tmpl.format(runner.epoch + 1)
        else:
            ckpt_filename = self.filename_tmpl.format(runner.iter + 1)

        if os.path.isdir(self.out_dir):
            out_file = os.path.join(self.out_dir, ckpt_filename)
            assert os.path.isfile(out_file), f'{out_file} does not exist'

            if self.task is None:
                self.task = self.clearml.Task.current_task()

            self.task.update_output_model(out_file)

