# crosskd_hooks.py

class SaveFeatureHook:
    def __init__(self):
        self.feat = None

    def __call__(self, module, input, output):
        # grad を保持した Tensor
        self.feat = output
