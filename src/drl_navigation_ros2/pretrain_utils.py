from typing import List
from tqdm import tqdm
import yaml


class Pretraining:
    def __init__(
        self,
        file_names: List[str],
        model: object,
        replay_buffer: object,
        reward_function,
    ):
        self.file_names = file_names
        self.model = model
        self.replay_buffer = replay_buffer
        self.reward_function = reward_function
        self.collision_count = 0

    def load_buffer(self):
        for file_name in self.file_names:
            print("Loading file: ", file_name)
            with open(file_name, "r") as file:
                samples = yaml.full_load(file)
                for i in tqdm(range(1, len(samples) - 1)):
                    sample = samples[i]
                    action = sample["action"]
                    collision = sample["collision"]
                    if collision:
                        self.collision_count += 1
                    cos = sample["cos"]
                    sin = sample["sin"]
                    distance = sample["distance"]
                    goal = sample["goal"]
                    latest_scan = sample["latest_scan"]
                    crash = False

                    state, terminal = self.model.prepare_state(
                        latest_scan, distance, cos, sin, goal, action, self.collision_count, crash  #latest_scan, distance, cos, sin, goal, a, collision_count, crash
                    )

                    if terminal:
                        self.collision_count = 0
                        continue

                    next_sample = samples[i + 1]
                    next_action = next_sample["action"]
                    next_collision = next_sample["collision"]
                    if next_collision:
                        self.collision_count += 1
                    next_cos = next_sample["cos"]
                    next_sin = next_sample["sin"]
                    next_distance = next_sample["distance"]
                    next_goal = next_sample["goal"]
                    next_latest_scan = next_sample["latest_scan"]
                    next_crash = False
                    next_state, next_terminal = self.model.prepare_state(
                        next_latest_scan,
                        next_distance,
                        next_cos,
                        next_sin,
                        next_goal,
                        next_action,
                        self.collision_count,
                        next_crash
                    )
                    reward = self.reward_function(
                        next_goal, next_collision, action, next_latest_scan, next_crash, next_cos
                    )
                    self.replay_buffer.add(
                        state, action, reward, next_terminal, next_state
                    )

        return self.replay_buffer

    def train(
        self,
        pretraining_iterations,
        replay_buffer,
        iterations,
        batch_size,
    ):
        print("Running Pretraining")
        for _ in tqdm(range(pretraining_iterations)):
            self.model.train(
                replay_buffer=replay_buffer,
                iterations=iterations,
                batch_size=batch_size,
            )
        print("Model Pretrained")
