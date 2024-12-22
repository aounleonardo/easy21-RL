# %%
import random
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from environment import ACTIONS, get_starting_state, step


class MCModel:
    N0 = 100

    def __init__(self) -> None:
        self.Q = np.zeros((22, 11, 2), dtype=np.float16)
        self.N = np.zeros((22, 11, 2))

    def run_episode(self):
        episode = []
        state = get_starting_state()
        reward = 0
        while not state["is_terminal"]:
            action = self.pick_action(state)
            episode.append((state, action))
            state, reward = step(state, ACTIONS[action])

        for seen_state, seen_action in episode:
            sa = seen_state["player"], seen_state["dealer"], seen_action
            self.N[sa] += 1
            self.Q[sa] += (reward - self.Q[sa]) / self.N[sa]

    def pick_action(self, state):
        epsilon = self.N0 / (self.N0 + self.N[state["player"], state["dealer"]].sum())
        if random.random() <= epsilon:
            return random.choice([0, 1])
        return self.Q[state["player"], state["dealer"]].argmax().item()


class SarsaLambdaModel:
    N0 = 100

    def __init__(self, 𝛌) -> None:
        self.𝛌 = 𝛌
        self.Q = np.zeros((22, 11, 2), dtype=np.float16)
        self.N = np.zeros((22, 11, 2))

    def run_episode(self):
        E = np.zeros((22, 11, 2))
        state = get_starting_state()
        action = self.pick_action(state)

        while not state["is_terminal"]:
            next_state, reward = step(state, ACTIONS[action])
            if next_state["is_terminal"]:
                next_action = None
                𝛅 = reward - self.Q[state["player"], state["dealer"], action]
            else:
                next_action = self.pick_action(next_state)
                𝛅 = (
                    reward
                    + self.Q[next_state["player"], next_state["dealer"], next_action]
                    - self.Q[state["player"], state["dealer"], action]
                )
            E[state["player"], state["dealer"], action] += 1
            self.N[state["player"], state["dealer"], action] += 1

            self.Q += np.divide(
                𝛅 * E, self.N, out=np.zeros_like(self.Q), where=self.N != 0
            )
            E *= self.λ

            state, action = next_state, next_action

    def pick_action(self, state):
        epsilon = self.N0 / (self.N0 + self.N[state["player"], state["dealer"]].sum())
        if random.random() <= epsilon:
            return random.choice([0, 1])
        return self.Q[state["player"], state["dealer"]].argmax().item()


def plot_metrics(model):
    cols, rows = 2, 2
    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(rows, cols, 1, projection="3d")
    ax.set_title("V*(s)")
    Q = model.Q[1:, 1:].max(-1)
    X, Y = np.meshgrid(range(Q.shape[1]), range(Q.shape[0]))
    ax.plot_wireframe(X, Y, Q, alpha=0.8)
    ax.set_xticks(range(Q.shape[1]), range(1, Q.shape[1] + 1))
    ax.set_yticks(range(0, Q.shape[0], 2), range(1, Q.shape[0] + 1, 2))
    ax.view_init(10, 330, 0)

    ax = fig.add_subplot(rows, cols, 2, projection=None)
    ax.set_title("𝛑(s)")
    plt.imshow(model.Q[1:, 1:].argmax(-1))
    plt.colorbar()
    ax.set_xticks(range(Q.shape[1]), range(1, Q.shape[1] + 1))
    ax.set_yticks(range(0, Q.shape[0], 2), range(1, Q.shape[0] + 1, 2))

    ax = fig.add_subplot(rows, cols, 3, projection="3d")
    ax.set_title("Q(s)")
    Q = model.Q[1:, 1:]
    X, Y = np.meshgrid(range(Q.shape[1]), range(Q.shape[0]))
    ax.plot_surface(X, Y, Q[:, :, 0], alpha=0.8, cmap="Oranges")
    ax.plot_surface(X, Y, Q[:, :, 1], alpha=0.8, cmap="Greens")
    ax.set_xticks(range(Q.shape[1]), range(1, Q.shape[1] + 1))
    ax.set_yticks(range(0, Q.shape[0], 2), range(1, Q.shape[0] + 1, 2))
    ax.view_init(10, 330, 0)

    ax = fig.add_subplot(rows, cols, 4, projection="3d")
    ax.set_title("N(s)")
    N = model.N[1:, 1:]
    X, Y = np.meshgrid(range(N.shape[1]), range(N.shape[0]))
    ax.plot_surface(X, Y, N[:, :, 0], alpha=0.8, cmap="Oranges")
    ax.plot_surface(X, Y, N[:, :, 1], alpha=0.8, cmap="Greens")
    ax.set_xticks(range(N.shape[1]), range(1, N.shape[1] + 1))
    ax.set_yticks(range(0, N.shape[0], 2), range(1, N.shape[0] + 1, 2))
    ax.view_init(10, 330, 0)
    plt.show()


if __name__ == "__main__":
    mc_model = MCModel()
    for _ in tqdm(range(1000000)):
        mc_model.run_episode()

    # %%
    td_models = {}
    learning_curves = {0.0: [], 1.0: []}
    for 𝛌 in tqdm(np.linspace(0, 1, 11)):
        td_models[𝛌] = SarsaLambdaModel(𝛌)
        for _ in range(1000):
            td_models[𝛌].run_episode()
            if 𝛌 not in learning_curves:
                continue
            learning_curves[𝛌].append(np.linalg.norm(td_models[𝛌].Q - mc_model.Q))

    errors = {𝛌: np.linalg.norm(model.Q - mc_model.Q) for 𝛌, model in td_models.items()}

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(121)
    ax.plot(errors.keys(), errors.values())
    ax.set_xticks(list(errors.keys()), [f"{𝛌:.1f}" for 𝛌 in errors])
    ax = plt.subplot(122)
    ax.plot(learning_curves[0], label="TD(0)")
    ax.plot(learning_curves[1], label="TD(1)")
    ax.legend()
    plt.show()

    # %%
