from types import SimpleNamespace

import pytest
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler

from src.cli.eval_i2v_euler import install_flowmatch_euler_scheduler


def test_install_flowmatch_euler_scheduler_preserves_flow_shift():
    original = UniPCMultistepScheduler(
        num_train_timesteps=1000,
        solver_order=2,
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        flow_shift=5.0,
    )
    pipe = SimpleNamespace(scheduler=original)

    flow_shift = install_flowmatch_euler_scheduler(pipe)

    assert flow_shift == 5.0
    assert isinstance(pipe.scheduler, FlowMatchEulerDiscreteScheduler)
    assert pipe.scheduler.config.shift == 5.0
    assert pipe.scheduler.config.stochastic_sampling is False


def test_install_flowmatch_euler_scheduler_falls_back_to_existing_shift():
    scheduler = SimpleNamespace(config={"num_train_timesteps": 1000, "shift": 3.0})
    pipe = SimpleNamespace(scheduler=scheduler)

    flow_shift = install_flowmatch_euler_scheduler(pipe)

    assert flow_shift == 3.0
    assert pipe.scheduler.config.shift == 3.0


def test_install_flowmatch_euler_scheduler_rejects_invalid_shift():
    scheduler = SimpleNamespace(config={"num_train_timesteps": 1000, "flow_shift": "invalid"})
    pipe = SimpleNamespace(scheduler=scheduler)

    with pytest.raises(ValueError):
        install_flowmatch_euler_scheduler(pipe)
