from candleprep.recovery.registry import REGISTRY
from candleprep.schema import RecoveryError


def execute(text, operations):
    for name in operations:
        if name not in REGISTRY:
            raise RecoveryError("Unknown operation: " + str(name))
        text = REGISTRY[name](text)
        if type(text) is not str:
            raise TypeError("Registered operations must preserve the CSV text payload")
    return text
