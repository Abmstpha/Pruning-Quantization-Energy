"""Model compression utilities (pruning and quantization)."""

from .pruning import (
    global_unstructured_prune,
    structured_channel_prune_resnet,
    structured_channel_prune_mixer,
    recalibrate_batch_norm
)
from .quantization import (
    apply_ptq_dynamic,
    replace_head_with_qat,
    warmup_qat_observers,
    QATLinear
)

__all__ = [
    'global_unstructured_prune',
    'structured_channel_prune_resnet', 
    'structured_channel_prune_mixer',
    'recalibrate_batch_norm',
    'apply_ptq_dynamic',
    'replace_head_with_qat',
    'warmup_qat_observers',
    'QATLinear'
]
