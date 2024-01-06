import jax
import jax.numpy as jnp
import numpy as np

import flax.linen as nn
from flax.training import train_state
import optax

from datasets import load_dataset
from PIL import Image


def reparameterize(mu: jnp.ndarray, logvar: jnp.ndarray, rng):
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(rng, mu.shape)
    return mu + eps * std


class Encoder(nn.Module):
    n_latent: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(512)(x)
        x = nn.silu(x)

        mu_x = nn.Dense(self.n_latent)(x)
        logvar_x = nn.Dense(self.n_latent)(x)

        return mu_x, logvar_x


class Decoder(nn.Module):
    n_latent: int

    @nn.compact
    def __call__(self, z: jnp.ndarray):
        z = nn.Dense(512)(z)
        z = nn.silu(z)
        return nn.Dense(28 * 28)(z) 


class VAE(nn.Module):
    n_latent: int

    def setup(self):
        self.encoder = Encoder(n_latent=self.n_latent)
        self.decoder = Decoder(n_latent=self.n_latent)

    def __call__(self, x: jnp.ndarray, rng):
        mu_x, logvar_x = self.encoder(x)
        z = reparameterize(mu_x, logvar_x, rng)

        recon = self.decode(z)
        return recon, mu_x, logvar_x

    def decode(self, z: jnp.ndarray):
        recon = self.decoder(z)
        recon = jnp.tanh(recon)
        return recon


@jax.jit
def apply_model(state: train_state.TrainState,
                x: jnp.ndarray,
                rng):
    def loss_fn(params):
        recon, mu, logvar = state.apply_fn({ "params": params }, x, rng)

        recon_loss = jnp.sum((x - recon) ** 2, axis=-1)
        kl_loss = -0.5 * jnp.sum(1 + logvar - mu ** 2 - jnp.exp(logvar), axis=-1)
        loss = jnp.mean(recon_loss + kl_loss)

        return loss, (recon_loss, kl_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, (recon_loss, kl_loss)), grads = grad_fn(state.params)

    return grads, recon_loss, kl_loss


@jax.jit
def update_model(state: train_state.TrainState, grads):
    return state.apply_gradients(grads=grads)


def create_train_state(rng):
    vae = VAE(n_latent=20)
    params = vae.init(rng, jnp.ones([1, 28 * 28]), rng=jax.random.PRNGKey(0))["params"]
    tx = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
    return train_state.TrainState.create(
        apply_fn=vae.apply,
        params=params,
        tx=tx
    )


@jax.jit
def preprocess(x: jnp.ndarray) -> jnp.ndarray:
    x = x.astype(jnp.float32) / 255. - 0.5
    x = x.reshape((-1, 28 * 28))
    return x


@jax.jit
def postprocess(x: jnp.ndarray) -> jnp.ndarray:
    x = x.reshape((-1, 28, 28))
    x = (x + 0.5) * 255.
    x = jnp.clip(x, 0, 255)
    x = x.astype(jnp.uint8)


def train_epoch(state: train_state.TrainState,
                ds, batch_size: int, rng):
    recon_losses = []
    kl_losses = []

    ds = ds.shuffle()
    for batch in ds.iter(batch_size=batch_size):
        x = batch["image"]
        x = preprocess(x)

        rng, subrng = jax.random.split(rng)
        grads, recon_loss, kl_loss = apply_model(state, x, rng=subrng)

        state = update_model(state, grads)

        recon_losses.append(recon_loss)
        kl_losses.append(kl_loss)

    recon_loss = np.mean(recon_loss)
    kl_loss = np.mean(kl_loss)

    return state, recon_loss, kl_loss


def sample_images(num_images: int, state: train_state.TrainState, rng):
    z = jax.random.normal(rng, (num_images, 20))
    x = state.apply_fn({"params": state.params}, z)
    x = postprocess(x)

    return x


def main():
    rng = jax.random.PRNGKey(42)
    init_rng, train_rng, eval_rng = jax.random.split(rng, 3)

    state = create_train_state(init_rng)

    ds = load_dataset("mnist")
    ds = ds.with_format("jax")

    for epoch in range(50):
        state, recon_loss, kl_loss = train_epoch(state, ds["train"], batch_size=128, rng=train_rng)
        print(f"Epoch {epoch} | Recon Loss: {recon_loss} | KL Loss: {kl_loss}")

    images = sample_images(10, state, rng=eval_rng)
    images = np.concatenate(images, axis=1)
    images = Image.fromarray(images)
    images.save(f"images/{epoch}.png")


if __name__ == "__main__":
    main()
