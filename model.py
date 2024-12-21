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

    plt.figure(figsize=(8, 12))
    cols, rows = 2, 3
    plt.subplot(rows, cols, 1)
    plt.title("V*(s)")
    plt.imshow(model.Q.max(axis=-1))
    plt.colorbar()
    plt.subplot(rows, cols, 2)
    plt.title("π(s)")
    plt.imshow(model.Q.argmax(-1))
    plt.colorbar()
    plt.subplot(rows, cols, 3)
    plt.title("Q(s, 0)")
    plt.imshow(model.Q[:,:,0])
    plt.colorbar()
    plt.subplot(rows, cols, 4)
    plt.title("Q(s, 1)")
    plt.imshow(model.Q[:,:,1])
    plt.colorbar()
    plt.subplot(rows, cols, 5)
    plt.title("N(s, 0)")
    plt.imshow(model.N[:, :, 0])
    plt.colorbar()
    plt.subplot(rows, cols, 6)
    plt.title("N(s, 1)")
    plt.imshow(model.N[:, :, 1])
    plt.colorbar()
    plt.show()

# %%
