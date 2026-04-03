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
        return self.state()

    def state(self):
        return Observation(
            ticket_id=self.current_ticket["id"],
            user_message=self.current_ticket["message"],
            category=None,
            priority=None,
            history=self.history,
            time_elapsed=self.steps
        )

    def step(self, action: Action):
        reward = 0.0
        done = False

        if action.action_type == "classify":
            self.predicted_category = action.content
        if action.content == self.current_ticket["true_category"]:
            reward += 0.2

        elif action.action_type == "prioritize":
            self.predicted_priority = action.content
            if action.content == self.current_ticket["true_priority"]:
                reward += 0.2

        elif action.action_type == "respond":
            if "refund" in (action.content or "").lower():
                reward += 0.3

        elif action.action_type == "resolve":
            reward += 0.5
            done = True
        
        reward -=0.05

        self.steps += 1
        if self.steps > 10:
            done = True
            reward -= 1.0

        self.predicted_category = None
        self.predicted_priority = None
        self.history.append(action.action_type)
        return self.state(), reward, done, {}