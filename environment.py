import random


ACTIONS = ["stick", "hit"]
COLOUR_MULTIPLIERS = {
    "black": 1,
    "red": -1,
}

def draw() -> tuple[int, str]:
    value = random.randint(1, 10)
    colour = random.choice(["red", "black", "black"])
    return value, colour

def get_starting_state() -> dict:
    return {
        "dealer": random.randint(1, 10),
        "player": random.randint(1, 10),
        "is_terminal": False,
    }

def step(state: dict, action: str) -> tuple[dict, int]:
    assert not state["is_terminal"]
    state = state.copy()
    match action:
        case "stick" | "s":
            while True:
                value, colour = draw()
                state["dealer"] += COLOUR_MULTIPLIERS[colour] * value
                if state["dealer"] > 21 or state["dealer"] < 1:
                    state["is_terminal"] = True
                    return state, 1
                elif state["dealer"] >= 17:
                    state["is_terminal"] = True
                    return state,  (state["player"] > state["dealer"]) - (state["player"] < state["dealer"])
        case "hit" | "h":
            value, colour = draw()
            state["player"] += COLOUR_MULTIPLIERS[colour] * value
            if state["player"] > 21 or state["player"] < 1:
                state["is_terminal"] = True
                return state, -1
            return state, 0
        case _:
            raise ValueError(f"Illegal action {action}")

def main():
    state = get_starting_state()
    while True:
        print(state)
        action = input("hit (h) or stick (s)?")
        state, reward = step(state, action)
        if reward != 0:
            print(f"{state = }; {reward = }")
            break


if __name__ == "__main__":
    main()
