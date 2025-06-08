import torch.amp.grad_scaler as ptags
import torch.nn.utils.clip_grad as ptnucg
import torch.optim as pto


Adam = pto.Adam


####


GradScaler = ptags.GradScaler


class ClipGradNorm:
    """"""

    def __init__(self, max_norm, norm_type=2):  # norm_type: 2 > inf
        self.max_norm = max_norm
        self.norm_type = norm_type

    def __call__(self, params):
        return ptnucg.clip_grad_norm_(params, self.max_norm, self.norm_type)


class ClipGradValue:
    """"""

    def __init__(self, max_value):
        self.max_value = max_value

    def __call__(self, params):
        return ptnucg.clip_grad_value_(params, self.max_value)
