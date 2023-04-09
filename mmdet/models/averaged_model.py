# Copyright (c) OpenMMLab. All rights reserved.
import copy
import itertools
import logging
import warnings
from abc import abstractmethod
from copy import deepcopy
from typing import Dict, Optional

import torch
import torch.nn as nn
from mmengine.hooks import Hook
from mmengine.hooks.hook import DATA_BATCH
from mmengine.logging import print_log
from mmengine.model import is_model_wrapper
from mmengine.registry import MODELS
from torch import Tensor

from mmdet.registry import HOOKS


@HOOKS.register_module()
class EMAHookV2(Hook):
    """A Hook to apply Exponential Moving Average (EMA) on the model during
    training.

    Note:
        - EMAHook takes priority over CheckpointHook.
        - The original model parameters are actually saved in ema field after
          train.
        - ``begin_iter`` and ``begin_epoch`` cannot be set at the same time.

    Args:
        ema_type (str): The type of EMA strategy to use. You can find the
            supported strategies in :mod:`mmengine.model.averaged_model`.
            Defaults to 'ExponentialMovingAverage'.
        strict_load (bool): Whether to strictly enforce that the keys of
            ``state_dict`` in checkpoint match the keys returned by
            ``self.module.state_dict``. Defaults to False.
            Changed in v0.3.0.
        begin_iter (int): The number of iteration to enable ``EMAHook``.
            Defaults to 0.
        begin_epoch (int): The number of epoch to enable ``EMAHook``.
            Defaults to 0.
        **kwargs: Keyword arguments passed to subclasses of
            :obj:`BaseAveragedModel`
    """

    priority = 'NORMAL'

    def __init__(self,
                 ema_type: str = 'ExponentialMovingAverage',
                 strict_load: bool = False,
                 begin_iter: int = 0,
                 begin_epoch: int = 0,
                 **kwargs):
        self.strict_load = strict_load
        self.ema_cfg = dict(type=ema_type, **kwargs)
        assert not (begin_iter != 0 and begin_epoch != 0), (
            '`begin_iter` and `begin_epoch` should not be both set.')
        assert begin_iter >= 0, (
            '`begin_iter` must larger than or equal to 0, '
            f'but got begin_iter: {begin_iter}')
        assert begin_epoch >= 0, (
            '`begin_epoch` must larger than or equal to 0, '
            f'but got begin_epoch: {begin_epoch}')
        self.begin_iter = begin_iter
        self.begin_epoch = begin_epoch
        # If `begin_epoch` and `begin_iter` are not set, `EMAHook` will be
        # enabled at 0 iteration.
        self.enabled_by_epoch = self.begin_epoch > 0

    def before_run(self, runner) -> None:
        """Create an ema copy of the model.

        Args:
            runner (Runner): The runner of the training process.
        """
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        self.src_model = model
        self.ema_model = MODELS.build(
            self.ema_cfg, default_args=dict(model=self.src_model))

    def before_train(self, runner) -> None:
        """Check the begin_epoch/iter is smaller than max_epochs/iters.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.enabled_by_epoch:
            assert self.begin_epoch <= runner.max_epochs, (
                'self.begin_epoch should be smaller than or equal to '
                f'runner.max_epochs: {runner.max_epochs}, but got '
                f'begin_epoch: {self.begin_epoch}')
        else:
            assert self.begin_iter <= runner.max_iters, (
                'self.begin_iter should be smaller than or equal to '
                f'runner.max_iters: {runner.max_iters}, but got '
                f'begin_iter: {self.begin_iter}')

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """Update ema parameter.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            outputs (dict, optional): Outputs from model. Defaults to None.
        """
        if self._ema_started(runner):
            self.ema_model.update_parameters(self.src_model)
        else:
            ema_params = self.ema_model.module.state_dict()
            src_params = self.src_model.state_dict()
            for k, p in ema_params.items():
                if src_params[k].shape != p.shape:
                    p = p.resize_(*src_params[k].shape)
                p.data.copy_(src_params[k].data)

    def before_val_epoch(self, runner) -> None:
        """We load parameter values from ema model to source model before
        validation.

        Args:
            runner (Runner): The runner of the training process.
        """
        self._swap_ema_parameters()

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """We recover source model's parameter from ema model after validation.

        Args:
            runner (Runner): The runner of the validation process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on validation dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        self._swap_ema_parameters()

    def before_test_epoch(self, runner) -> None:
        """We load parameter values from ema model to source model before test.

        Args:
            runner (Runner): The runner of the training process.
        """
        self._swap_ema_parameters()

    def after_test_epoch(self,
                         runner,
                         metrics: Optional[Dict[str, float]] = None) -> None:
        """We recover source model's parameter from ema model after test.

        Args:
            runner (Runner): The runner of the testing process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on test dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        self._swap_ema_parameters()

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        """Save ema parameters to checkpoint.

        Args:
            runner (Runner): The runner of the testing process.
        """
        checkpoint['ema_state_dict'] = self.ema_model.state_dict()
        # Save ema parameters to the source model's state dict so that we
        # can directly load the averaged model weights for deployment.
        # Swapping the state_dict key-values instead of swapping model
        # parameters because the state_dict is a shallow copy of model
        # parameters.
        self._swap_ema_state_dict(checkpoint)

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        """Resume ema parameters from checkpoint.

        Args:
            runner (Runner): The runner of the testing process.
        """
        from mmengine.runner.checkpoint import load_state_dict
        if 'ema_state_dict' in checkpoint and runner._resume:
            # The original model parameters are actually saved in ema
            # field swap the weights back to resume ema state.
            self._swap_ema_state_dict(checkpoint)
            self.ema_model.load_state_dict(
                checkpoint['ema_state_dict'], strict=self.strict_load)

        # Support load checkpoint without ema state dict.
        else:
            if runner._resume:
                print_log(
                    'There is no `ema_state_dict` in checkpoint. '
                    '`EMAHook` will make a copy of `state_dict` as the '
                    'initial `ema_state_dict`', 'current', logging.WARNING)
            load_state_dict(
                self.ema_model.module,
                copy.deepcopy(checkpoint['state_dict']),
                strict=self.strict_load)

    def _swap_ema_parameters(self) -> None:
        """Swap the parameter of model with ema_model."""
        avg_param = (
            itertools.chain(self.ema_model.module.parameters(),
                            self.ema_model.module.buffers())
            if self.ema_model.update_buffers else
            self.ema_model.module.parameters())
        src_param = (
            itertools.chain(self.src_model.parameters(),
                            self.src_model.buffers())
            if self.ema_model.update_buffers else self.src_model.parameters())
        for p_avg, p_src in zip(avg_param, src_param):
            tmp = p_avg.data.clone()
            if p_src.shape != p_avg.shape:
                p_avg = p_avg.resize_(*p_src.shape)
            p_avg.data.copy_(p_src.data)
            if p_src.shape != tmp.shape:
                p_src = p_src.resize_(*tmp.shape)
            p_src.data.copy_(tmp)

    def _swap_ema_state_dict(self, checkpoint):
        """Swap the state dict values of model with ema_model."""
        model_state = checkpoint['state_dict']
        ema_state = checkpoint['ema_state_dict']
        for k in ema_state:
            if k[:7] == 'module.':
                tmp = ema_state[k]
                ema_state[k] = model_state[k[7:]]
                model_state[k[7:]] = tmp

    def _ema_started(self, runner) -> bool:
        """Whether ``EMAHook`` has been initialized at current iteration or
        epoch.

        :attr:`ema_model` will be initialized when ``runner.iter`` or
        ``runner.epoch`` is greater than ``self.begin`` for the first time.

        Args:
            runner (Runner): Runner of the training, validation process.

        Returns:
            bool: Whether ``EMAHook`` has been initialized.
        """
        if self.enabled_by_epoch:
            return runner.epoch + 1 >= self.begin_epoch
        else:
            return runner.iter + 1 >= self.begin_iter


class BaseAveragedModelV2(nn.Module):
    """A base class for averaging model weights.

    Weight averaging, such as SWA and EMA, is a widely used technique for
    training neural networks. This class implements the averaging process
    for a model. All subclasses must implement the `avg_func` method.
    This class creates a copy of the provided module :attr:`model`
    on the :attr:`device` and allows computing running averages of the
    parameters of the :attr:`model`.

    The code is referenced from: https://github.com/pytorch/pytorch/blob/master/torch/optim/swa_utils.py.

    Different from the `AveragedModel` in PyTorch, we use in-place operation
    to improve the parameter updating speed, which is about 5 times faster
    than the non-in-place version.

    In mmengine, we provide two ways to use the model averaging:

    1. Use the model averaging module in hook:
       We provide an :class:`mmengine.hooks.EMAHook` to apply the model
       averaging during training. Add ``custom_hooks=[dict(type='EMAHook')]``
       to the config or the runner.

    2. Use the model averaging module directly in the algorithm. Take the ema
       teacher in semi-supervise as an example:

       >>> from mmengine.model import ExponentialMovingAverage
       >>> student = ResNet(depth=50)
       >>> # use ema model as teacher
       >>> ema_teacher = ExponentialMovingAverage(student)

    Args:
        model (nn.Module): The model to be averaged.
        interval (int): Interval between two updates. Defaults to 1.
        device (torch.device, optional): If provided, the averaged model will
            be stored on the :attr:`device`. Defaults to None.
        update_buffers (bool): if True, it will compute running averages for
            both the parameters and the buffers of the model. Defaults to
            False.
    """  # noqa: E501

    def __init__(self,
                 model: nn.Module,
                 interval: int = 1,
                 device: Optional[torch.device] = None,
                 update_buffers: bool = False) -> None:
        super().__init__()
        self.module = deepcopy(model).requires_grad_(False)
        self.interval = interval
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer('steps',
                             torch.tensor(0, dtype=torch.long, device=device))
        self.update_buffers = update_buffers
        if update_buffers:
            self.avg_parameters = self.module.state_dict()
        else:
            self.avg_parameters = dict(self.module.named_parameters())

    @abstractmethod
    def avg_func(self, averaged_param: Tensor, source_param: Tensor,
                 steps: int) -> None:
        """Use in-place operation to compute the average of the parameters. All
        subclasses must implement this method.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """

    def forward(self, *args, **kwargs):
        """Forward method of the averaged model."""
        return self.module(*args, **kwargs)

    def update_parameters(self, model: nn.Module) -> None:
        """Update the parameters of the model. This method will execute the
        ``avg_func`` to compute the new parameters and update the model's
        parameters.

        Args:
            model (nn.Module): The model whose parameters will be averaged.
        """
        src_parameters = (
            model.state_dict()
            if self.update_buffers else dict(model.named_parameters()))
        if self.steps == 0:
            for k, p_avg in self.avg_parameters.items():
                if src_parameters[k].shape != p_avg.shape:
                    p_avg = p_avg.resize_(*src_parameters[k].shape)
                p_avg.data.copy_(src_parameters[k].data)
        elif self.steps % self.interval == 0:
            for k, p_avg in self.avg_parameters.items():
                if p_avg.dtype.is_floating_point:
                    device = p_avg.device
                    if src_parameters[k].shape != p_avg.shape:
                        p_avg = p_avg.resize_(*src_parameters[k].shape)
                        p_avg.data.copy_(src_parameters[k].data.to(device))
                    else:
                        self.avg_func(p_avg.data,
                                      src_parameters[k].data.to(device),
                                      self.steps)
        if not self.update_buffers:
            # If not update the buffers,
            # keep the buffers in sync with the source model.
            for b_avg, b_src in zip(self.module.buffers(), model.buffers()):
                b_avg.data.copy_(b_src.data.to(b_avg.device))
        self.steps += 1


@MODELS.register_module()
class StochasticWeightAverageV2(BaseAveragedModelV2):
    """Implements the stochastic weight averaging (SWA) of the model.

    Stochastic Weight Averaging was proposed in `Averaging Weights Leads to
    Wider Optima and Better Generalization, UAI 2018.
    <https://arxiv.org/abs/1803.05407>`_ by Pavel Izmailov, Dmitrii
    Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson.
    """

    def avg_func(self, averaged_param: Tensor, source_param: Tensor,
                 steps: int) -> None:
        """Compute the average of the parameters using stochastic weight
        average.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """
        averaged_param.add_(
            source_param - averaged_param,
            alpha=1 / float(steps // self.interval + 1))


@MODELS.register_module()
class ExponentialMovingAverageV2(BaseAveragedModelV2):
    r"""Implements the exponential moving average (EMA) of the model.

    All parameters are updated by the formula as below:

        .. math::

            Xema_{t+1} = (1 - momentum) * Xema_{t} +  momentum * X_t

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically,
        :math:`Xema_{t+1}` is the moving average and :math:`X_t` is the
        new observed value. The value of momentum is usually a small number,
        allowing observed values to slowly update the ema parameters.

    Args:
        model (nn.Module): The model to be averaged.
        momentum (float): The momentum used for updating ema parameter.
            Defaults to 0.0002.
            Ema's parameter are updated with the formula
            :math:`averaged\_param = (1-momentum) * averaged\_param +
            momentum * source\_param`.
        interval (int): Interval between two updates. Defaults to 1.
        device (torch.device, optional): If provided, the averaged model will
            be stored on the :attr:`device`. Defaults to None.
        update_buffers (bool): if True, it will compute running averages for
            both the parameters and the buffers of the model. Defaults to
            False.
    """  # noqa: W605

    def __init__(self,
                 model: nn.Module,
                 momentum: float = 0.0002,
                 interval: int = 1,
                 device: Optional[torch.device] = None,
                 update_buffers: bool = False) -> None:
        super().__init__(model, interval, device, update_buffers)
        assert 0.0 < momentum < 1.0, 'momentum must be in range (0.0, 1.0)'\
                                     f'but got {momentum}'
        if momentum > 0.5:
            warnings.warn(
                'The value of momentum in EMA is usually a small number,'
                'which is different from the conventional notion of '
                f'momentum but got {momentum}. Please make sure the '
                f'value is correct.')
        self.momentum = momentum

    def avg_func(self, averaged_param: Tensor, source_param: Tensor,
                 steps: int) -> None:
        """Compute the moving average of the parameters using exponential
        moving average.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """
        averaged_param.lerp_(source_param, self.momentum)


@MODELS.register_module()
class MomentumAnnealingEMAV2(ExponentialMovingAverageV2):
    r"""Exponential moving average (EMA) with momentum annealing strategy.

    Args:
        model (nn.Module): The model to be averaged.
        momentum (float): The momentum used for updating ema parameter.
            Defaults to 0.0002.
            Ema's parameter are updated with the formula
            :math:`averaged\_param = (1-momentum) * averaged\_param +
            momentum * source\_param`.
        gamma (int): Use a larger momentum early in training and gradually
            annealing to a smaller value to update the ema model smoothly. The
            momentum is calculated as max(momentum, gamma / (gamma + steps))
            Defaults to 100.
        interval (int): Interval between two updates. Defaults to 1.
        device (torch.device, optional): If provided, the averaged model will
            be stored on the :attr:`device`. Defaults to None.
        update_buffers (bool): if True, it will compute running averages for
            both the parameters and the buffers of the model. Defaults to
            False.
    """

    def __init__(self,
                 model: nn.Module,
                 momentum: float = 0.0002,
                 gamma: int = 100,
                 interval: int = 1,
                 device: Optional[torch.device] = None,
                 update_buffers: bool = False) -> None:
        super().__init__(
            model=model,
            momentum=momentum,
            interval=interval,
            device=device,
            update_buffers=update_buffers)
        assert gamma > 0, f'gamma must be greater than 0, but got {gamma}'
        self.gamma = gamma

    def avg_func(self, averaged_param: Tensor, source_param: Tensor,
                 steps: int) -> None:
        """Compute the moving average of the parameters using the linear
        momentum strategy.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """
        momentum = max(self.momentum, self.gamma / (self.gamma + self.steps))
        averaged_param.lerp_(source_param, momentum)
