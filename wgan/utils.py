import torch
from torchvision.utils import save_image

def gradient_penalty(critic, real, fake, device):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)

    interpolated = real * epsilon + fake * (1 - epsilon)
    interpolated.requires_grad_(True)

    scores = critic(interpolated)

    grad = torch.autograd.grad(
        inputs=interpolated,
        outputs=scores,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True
    )[0]

    grad = grad.view(batch_size, -1)
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


def save_samples(generator, z_dim, device, filename):
    generator.eval()
    z = torch.randn(64, z_dim, 1, 1, device=device)
    samples = generator(z)
    save_image(samples, filename, nrow=8, normalize=True)
    generator.train()
