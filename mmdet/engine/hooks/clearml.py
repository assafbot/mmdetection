from typing import Optional, Dict

from mmengine.dist import master_only
from mmengine.hooks import Hook

from mmdet.registry import HOOKS


@HOOKS.register_module()
class ClearMLLoggerHook(Hook):
    def __init__(self, init_kwargs: Optional[Dict] = None) -> None:
        self.clearml = None
        self.task = None
        self.init_kwargs = init_kwargs
        self.import_clearml()

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
        self.task = self.clearml.Task.init(**task_kwargs)
