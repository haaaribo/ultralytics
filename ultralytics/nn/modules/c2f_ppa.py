# ultralytics/nn/modules/c2f_ppa.py
import torch, torch.nn as nn

__all__ = ("C2f_PPA",)

class C2f_PPA(nn.Module):
    """
    Ultralytics C2f 시그니처(c1,c2,n,shortcut) + patch_size 옵션
    """
    def __init__(self, c1, c2, n=1, shortcut=True, patch_size=4):
        super().__init__()
        self.patch_size = patch_size

        # Local branch
        self.local = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, 1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )

        # Global branch
        self.global_proj = nn.Conv2d(c1, c2, patch_size, patch_size, bias=False)
        self.global_norm = nn.LayerNorm(c2)
        self.global_mlp  = nn.Sequential(
            nn.Linear(c2, c2),
            nn.GELU(),
            nn.Linear(c2, c2)
        )

        # Point-wise fusion
        self.pointwise = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn        = nn.BatchNorm2d(c2)
        self.act       = nn.ReLU(inplace=True)

    def forward(self, x):
        B,_,H,W = x.shape
        loc = self.local(x)
        g   = self.global_proj(x).mean(dim=[2,3])
        g   = self.global_mlp(self.global_norm(g)).view(B,-1,1,1)
        pw  = self.pointwise(x)
        out = loc + pw + g
        return self.act(self.bn(out))