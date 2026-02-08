# kd_hooks.py
from utils import log_feat_stats

class FeatureHook:
    def __init__(self):
        self.feature = None

    def __call__(self, module, input, output):
        self.feature = output


class FeatureVisualizeHook:
    def __init__(self, path):
        self.feature = None
        self.path = path

    def __call__(self, module, input, output):
        self.feature = output
        log_feat_stats(self.feature, -1, -1, path=self.path)
