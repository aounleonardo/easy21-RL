# %%
import random
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import itertools as it
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
        𝛆 = self.N0 / (self.N0 + self.N[state["player"], state["dealer"]].sum())
        if random.random() <= 𝛆:
            return random.choice([0, 1])
        return self.Q[state["player"], state["dealer"]].argmax().item()

# %%
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
        𝛆 = self.N0 / (self.N0 + self.N[state["player"], state["dealer"]].sum())
        if random.random() <= 𝛆:
            return random.choice([0, 1])
        return self.Q[state["player"], state["dealer"]].argmax().item()


# %%

CUBOID_INTERVALS = {
    "dealer": [[1, 4], [4, 7], [7, 10]],
    "player": [[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]],
}


def build_𝛟():
    def build(state, action):
        features = np.zeros((6, 3, 2), dtype=np.bool)
        for x, (player_start, player_end) in enumerate(
            CUBOID_INTERVALS["player"]
        ):
            for y, (dealer_start, dealer_end) in enumerate(CUBOID_INTERVALS["dealer"]):
                features[x, y, action] = (
                    dealer_start <= state["dealer"] <= dealer_end
                    and player_start <= state["player"] <= player_end
                )
        return features.reshape(-1)
    
    ret = np.zeros((22, 11, 2, 36), dtype=np.float16)
    for player, dealer, action in it.product(
        range(1, 22),
        range(1, 11),
        range(2),
    ):
        ret[player, dealer, action] = build({"dealer": dealer, "player": player}, action)
    return ret

class LinearApproximationModel:
    𝛜 = 0.05
    𝚨 = 0.01
    𝛟 = build_𝛟()

    def __init__(self, 𝛌) -> None:
        self.𝛌 = 𝛌
        self.𝛉 = np.zeros((36), dtype=np.float16)

    def run_episode(self):
        E = np.zeros((36))
        state = get_starting_state()
        action = self.pick_action(state)
        features = self.𝛟[state["player"], state["dealer"], action]

        while not state["is_terminal"]:
            next_state, reward = step(state, ACTIONS[action])
            if next_state["is_terminal"]:
                next_action = next_features = None
                𝛅 = reward - np.dot(features, self.θ)
            else:
                next_action = self.pick_action(next_state)
                next_features = self.𝛟[next_state["player"], next_state["dealer"], next_action]
                𝛅 = reward + np.dot(next_features, self.θ) - np.dot(features, self.θ)
            # NOTE features is simply the gradient of q̂(S, A, w) with respect to w
            E = self.λ * E + features
            self.θ += self.𝚨 * 𝛅 * E

            state, action, features = next_state, next_action, next_features

    def pick_action(self, state):
        if random.random() <= self.ε:
            return random.choice([0, 1])
        return self.Q[state["player"], state["dealer"]].argmax().item()

    @property
    def Q(self):
        return np.dot(self.𝛟, self.θ)

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
    ax.set_title("π(s)")
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

    try:
        N = model.N[1:, 1:]
        ax = fig.add_subplot(rows, cols, 4, projection="3d")
        ax.set_title("N(s)")
        X, Y = np.meshgrid(range(N.shape[1]), range(N.shape[0]))
        ax.plot_surface(X, Y, N[:, :, 0], alpha=0.8, cmap="Oranges")
        ax.plot_surface(X, Y, N[:, :, 1], alpha=0.8, cmap="Greens")
        ax.set_xticks(range(N.shape[1]), range(1, N.shape[1] + 1))
        ax.set_yticks(range(0, N.shape[0], 2), range(1, N.shape[0] + 1, 2))
        ax.view_init(10, 330, 0)
    except AttributeError:
        ax = fig.add_subplot(rows, cols, 4, projection=None)
        ax.set_title("θ(s)")
        plt.imshow(model.𝛉.reshape(6, 6))
        plt.colorbar()

    plt.show()

# %%

if __name__ == "__main__":
    mc_model = MCModel()
    for _ in tqdm(range(1000000)):
        mc_model.run_episode()

    # %%
    
    td_models = {}
    td_learning_curves = {0.0: [], 1.0: []}
    for 𝛌 in tqdm(np.linspace(0, 1, 11)):
        td_models[𝛌] = SarsaLambdaModel(𝛌)
        for _ in range(10000):
            td_models[𝛌].run_episode()
            if 𝛌 not in td_learning_curves:
                continue
            td_learning_curves[𝛌].append(np.pow(td_models[𝛌].Q - mc_model.Q, 2).mean())

    la_models = {}
    la_learning_curves = {0.0: [], 1.0: []}
    for 𝛌 in tqdm(np.linspace(0, 1, 11)):
        la_models[𝛌] = LinearApproximationModel(𝛌)
        for _ in range(10000):
            la_models[𝛌].run_episode()
            if 𝛌 not in la_learning_curves:
                continue
            la_learning_curves[𝛌].append(np.pow(la_models[𝛌].Q - mc_model.Q, 2).mean())
    
    td_errors = {𝛌: np.pow(model.Q - mc_model.Q, 2).mean() for 𝛌, model in td_models.items()}
    la_errors = {𝛌: np.pow(model.Q - mc_model.Q, 2).mean() for 𝛌, model in la_models.items()}

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(121)
    ax.plot(td_errors.keys(), td_errors.values(), label="TD errors")
    ax.plot(la_errors.keys(), la_errors.values(), label="LA errors")
    ax.set_xticks(list(td_errors.keys()), [f"{𝛌:.1f}" for 𝛌 in td_errors])
    ax.legend()
    ax = plt.subplot(122)
    ax.plot(td_learning_curves[0], label="TD(0)")
    ax.plot(td_learning_curves[1], label="TD(1)")
    ax.plot(la_learning_curves[0], label="LA(0)")
    ax.plot(la_learning_curves[1], label="LA(1)")
    ax.legend()
    plt.show()

    # %%
