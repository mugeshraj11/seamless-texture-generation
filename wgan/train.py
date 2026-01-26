import torch
from tqdm import tqdm

from config import *
from dataset import get_dataloader
from model import Generator, Critic
from utils import gradient_penalty, save_samples

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

loader = get_dataloader(DATA_DIR, IMAGE_SIZE, BATCH_SIZE)

gen = Generator().to(device)
critic = Critic().to(device)

opt_gen = torch.optim.Adam(gen.parameters(), lr=LR, betas=(0.0, 0.9))
opt_critic = torch.optim.Adam(critic.parameters(), lr=LR, betas=(0.0, 0.9))

z = torch.randn(1, Z_DIM, 1, 1, device=device)
fake = gen(z)
print("Generator output shape:", fake.shape)

for epoch in tqdm(range(EPOCHS)):
    for real, _ in loader:
        real = real.to(device)

        # Train Critic
        for _ in range(CRITIC_ITERS):
            z = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)
            fake = gen(z)

            critic_real = critic(real)
            critic_fake = critic(fake.detach())

            gp = gradient_penalty(critic, real, fake, device)
            loss_critic = (
                -(critic_real.mean() - critic_fake.mean())
                + LAMBDA_GP * gp
            )

            opt_critic.zero_grad()
            loss_critic.backward()
            opt_critic.step()

        # Train Generator
        z = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)
        fake = gen(z)
        loss_gen = -critic(fake).mean()

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch} | G: {loss_gen:.3f} | C: {loss_critic:.3f}")

        torch.save({
        "epoch": epoch,
        "generator": gen.state_dict(),
        "critic": critic.state_dict(),
        "opt_gen": opt_gen.state_dict(),
        "opt_critic": opt_critic.state_dict(),
    }, f"wgan/checkpoints/wgan_epoch_{epoch}.pth")

