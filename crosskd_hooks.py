# crosskd_hooks.py
from utils import log_feat_stats

class SaveFeatureHook:
    def __init__(self):
        self.feat = None

    def __call__(self, module, input, output):
        # grad を保持した Tensor
        self.feat = output


class VisualizeFeatureHook:
    def __init__(self, path):
        self.feat = None
        self.path = path

    def __call__(self, module, input, output):
        # grad を保持した Tensor
        self.feat = output

        log_feat_stats(self.feat, -1, -1, path=self.path)
        
