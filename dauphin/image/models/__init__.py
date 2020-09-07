from dauphin.image.models.mlp import MLP
from dauphin.image.models.pyramidnet import PyramidNet
from dauphin.image.models.shake_shake import ShakeShake
from dauphin.image.models.wide_resnet import WideResNet
from dauphin.image.models.resnet import ResNet

ALL_MODELS = {
    "resnet": ResNet,
    "wide_resnet": WideResNet,
    "mlp": MLP,
    "pyramidnet": PyramidNet,
    "shake_shake": ShakeShake,
}
