# Workaround: In some PyTorch versions, DeviceMesh.size() passes string
# dimension names directly to _MeshLayout.__getitem__ which only accepts int
# indices, causing TypeError.  Patch size() to convert name → index first.
from torch.distributed.device_mesh import DeviceMesh as _DeviceMesh

_orig_device_mesh_size = _DeviceMesh.size


def _device_mesh_size_patched(self, mesh_dim=None):
    if isinstance(mesh_dim, str) and self.mesh_dim_names:
        mesh_dim = list(self.mesh_dim_names).index(mesh_dim)
    return _orig_device_mesh_size(self, mesh_dim)


_DeviceMesh.size = _device_mesh_size_patched

from src.trainer.base_grpo_trainer import BaseGRPOTrainer  # noqa: E402
from src.trainer.base_rl_trainer import BaseRLTrainer  # noqa: E402
from src.trainer.checkpoint import TrainState  # noqa: E402
from src.trainer.config import RLConfig, SFTConfig, TrainConfig  # noqa: E402
from src.trainer.dancegrpo_trainer import DanceGRPOTrainer  # noqa: E402
from src.trainer.i2v_trainer import I2VTrainer  # noqa: E402

__all__ = [
    "BaseGRPOTrainer",
    "BaseRLTrainer",
    "DanceGRPOTrainer",
    "I2VTrainer",
    "RLConfig",
    "SFTConfig",
    "TrainConfig",
    "TrainState",
]
