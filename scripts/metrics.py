import torch


def l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(x - y))


def psnr(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = torch.mean((x - y) ** 2)
    mse = torch.clamp(mse, min=1e-12)
    return 10.0 * torch.log10(torch.tensor(max_val * max_val, device=x.device) / mse)


def ssim_global(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Fast global SSIM approximation over batch/channels.

    x, y in [0,1], shape (N,C,H,W)
    """
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    mu_x = x.mean(dim=(-1, -2), keepdim=True)
    mu_y = y.mean(dim=(-1, -2), keepdim=True)

    sigma_x = ((x - mu_x) ** 2).mean(dim=(-1, -2), keepdim=True)
    sigma_y = ((y - mu_y) ** 2).mean(dim=(-1, -2), keepdim=True)
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=(-1, -2), keepdim=True)

    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    ssim_map = num / (den + 1e-12)
    return ssim_map.mean()
