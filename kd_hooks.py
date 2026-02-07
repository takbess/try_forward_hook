# kd_hooks.py

class FeatureHook:
    def __init__(self):
        self.feature = None

    def __call__(self, module, input, output):
        self.feature = output
