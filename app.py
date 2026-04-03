from env.environment import SmartOpsEnv
from env.models import Action
from env.tasks import easy_task


env = SmartOpsEnv()
obs = env.reset()
score = easy_task(env)
print("Task Score:", score)

print("Initial:", obs)

action = Action(action_type="classify", content="delivery")
obs, reward, done, _ = env.step(action)

print("After Step:", obs)
print("Reward:", reward)