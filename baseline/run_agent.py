from env.environment import SmartOpsEnv
from env.models import Action


def smart_agent(obs):
    text = obs.user_message.lower()

    if "refund" in text or "damaged" in text:
        return Action(action_type="classify", content="refund")

    if "late" in text or "delivery" in text:
        return Action(action_type="classify", content="delivery")

    if "payment" in text or "charged" in text:
        return Action(action_type="classify", content="billing")

    if "urgent" in text or "asap" in text or "immediately" in text:
        return Action(action_type="prioritize", content="high")

    return Action(
        action_type="respond",
        content="Sorry for the inconvenience. We are working on resolving your issue."
    )

def run():
    env = SmartOpsEnv()
    obs = env.reset()
    total_reward = 0

    print("Starting SmartOps Agent\n")

    for step in range(10):
        print(f"--- Step {step + 1} ---")
        print("Ticket:", obs.user_message)

        action = smart_agent(obs)

        obs, reward, done, _ = env.step(action)
        total_reward += reward

        print("Action:", action)
        print("Reward:", reward)
        print()

        if done:
            print("Task Completed")
            break

    print("Final Score:", total_reward)

if __name__ == "__main__":
    run()