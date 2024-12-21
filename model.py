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
        try:
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
        except IndexError:
            print(episode, state)
            raise
    
    def pick_action(self, state):
        epsilon = self.N0 / (self.N0 + self.N[state["player"], state["dealer"]].sum())
        if random.random() <= epsilon:
            return random.choice([0, 1])
        return self.Q[state["player"], state["dealer"]].argmax().item()

if __name__ == "__main__":
    model = MCModel()
# %%
    for _ in tqdm(range(1_000_000)):
        model.run_episode()

    cols, rows = 2, 2
    fig = plt.figure(figsize=(10, 8))
    
    ax = fig.add_subplot(rows, cols, 1, projection='3d')
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

    ax = fig.add_subplot(rows, cols, 3, projection='3d')
    ax.set_title("Q(s)")
    Q = model.Q[1:, 1:]
    X, Y = np.meshgrid(range(Q.shape[1]), range(Q.shape[0]))
    ax.plot_surface(X, Y, Q[:,:,0], alpha=0.8, cmap="Oranges")
    ax.plot_surface(X, Y, Q[:,:,1], alpha=0.8, cmap="Greens")
    ax.set_xticks(range(Q.shape[1]), range(1, Q.shape[1] + 1))
    ax.set_yticks(range(0, Q.shape[0], 2), range(1, Q.shape[0] + 1, 2))
    ax.view_init(10, 330, 0)

    ax = fig.add_subplot(rows, cols, 4, projection='3d')
    ax.set_title("N(s)")
    N = model.N[1:, 1:]
    X, Y = np.meshgrid(range(N.shape[1]), range(N.shape[0]))
    ax.plot_surface(X, Y, N[:,:,0], alpha=0.8, cmap="Oranges")
    ax.plot_surface(X, Y, N[:,:,1], alpha=0.8, cmap="Greens")
    ax.set_xticks(range(N.shape[1]), range(1, N.shape[1] + 1))
    ax.set_yticks(range(0, N.shape[0], 2), range(1, N.shape[0] + 1, 2))
    ax.view_init(10, 330, 0)
    plt.show()

# %%
