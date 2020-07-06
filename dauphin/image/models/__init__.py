from dauphin.image.models.mlp import MLP
from dauphin.image.models.pyramidnet import PyramidNet
from dauphin.image.models.shake_shake import ShakeShake
from dauphin.image.models.wide_resnet import WideResNet

ALL_MODELS = {
    "wide_resnet": WideResNet,
    "mlp": MLP,
    "pyramidnet": PyramidNet,
    "shake_shake": ShakeShake,
}
