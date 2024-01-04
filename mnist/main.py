import jax
import jax.numpy as jnp

import flax.linen as nn
from flax.training import train_state
import optax

import numpy as np

from datasets import load_dataset


class Net(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.float32) / 255. - 0.5
        x = x.reshape((-1, 28 * 28))

        x = nn.Dense(128)(x)
        x = nn.silu(x)
        x = nn.Dense(10)(x)
        return x


@jax.jit
def apply_model(state, x, y):
    def loss_fn(params):
        logits = state.apply_fn({ "params": params }, x)
        one_hot = jax.nn.one_hot(y, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def create_train_state(rng):
    net = Net()
    params = net.init(rng, jnp.ones([1, 28 * 28]))["params"]
    tx = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
    return train_state.TrainState.create(apply_fn=net.apply, params=params, tx=tx)


def train_epoch(state, ds, batch_size):
    losses = []
    accuracies = []

    ds = ds.shuffle()
    for batch in ds.iter(batch_size=batch_size):
        x, y = batch["image"], batch["label"]

        grad, loss, accuracy = apply_model(state, x, y)
        state = update_model(state, grad)

        losses.append(loss)
        accuracies.append(accuracy)

    loss = np.mean(losses)
    accuracy = np.mean(accuracies)

    return state, loss, accuracy


def main():
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)

    state = create_train_state(init_rng)

    ds = load_dataset("mnist")
    ds = ds.with_format("jax")

    for epoch in range(10):
        state, loss, accuracy = train_epoch(state, ds["train"], batch_size=128)
        print(f"epoch: {epoch}, loss: {loss:.4f}, accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
