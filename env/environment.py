import json
import random
from .models import Observation, Action

class SmartOpsEnv:
    
    def __init__(self):
        with open("data/tickets.json") as f:
            self.dataset = json.load(f)
        self.reset()

    def reset(self):
        self.current_ticket = random.choice(self.dataset)
        self.steps = 0
        self.history = []
        self.category = None
        self.priority = None
        return self.state()

    def state(self):
        return Observation(
            ticket_id=self.current_ticket["id"],
            user_message=self.current_ticket["message"],
            category=self.category,
            priority=self.priority,
            history=self.history,
            time_elapsed=self.steps
        )

    def step(self, action: Action):
        reward = 0.0
        done = False

        if action.action_type == "classify":
            if self.category is None:
                self.category = action.content
                if action.content == self.current_ticket["true_category"]:
                    reward += 0.3
                else:
                    reward -= 0.1
            else:
                reward -= 0.2

        elif action.action_type == "prioritize":
            if self.priority is None:
                self.priority = action.content
                if action.content == self.current_ticket["true_priority"]:
                    reward += 0.2
                else:
                    reward -= 0.1
            else:
                reward -= 0.2

        elif action.action_type == "respond":
            self.history.append(action.content)
            reward += 0.1

        elif action.action_type == "resolve":
            if self.category and self.priority:
                reward += 0.5
                done = True
            else:
                reward -= 0.3

        else:
            reward -= 0.2

        self.steps += 1
        reward -= 0.05

        if self.steps >= 10:
            done = True
            reward -= 1.0

        return self.state(), reward, done, {}