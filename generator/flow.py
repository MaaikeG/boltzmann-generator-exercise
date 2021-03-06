import torch 


class Flow(torch.nn.Module):
    """A normalizing flow made up from multiple invertible blocks.

    Parameters
    ----------
    blocks : list
        A list of invertible blocks
    """
    def __init__(self, blocks):
        super().__init__()
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, samples):
        total_jac_det_log = 0
        for block in self.blocks:
            samples, jac_det_log = block.forward(samples)
            total_jac_det_log += jac_det_log
        return samples, total_jac_det_log

    def inverse(self, samples):
        total_jac_det_log = 0
        for block in reversed(self.blocks):
            samples, jac_det_log = block.inverse(samples)
            total_jac_det_log += jac_det_log
        return samples, total_jac_det_log
