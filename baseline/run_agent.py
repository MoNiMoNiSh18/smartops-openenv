from env.environment import SmartOpsEnv
from env.models import Action


def classify_ticket(text):
    text = text.lower()

    if "refund" in text or "damaged" in text or "poor" in text or "wrong" in text:
        return "refund"

    if "payment" in text or "charged" in text or "invoice" in text or "purchase" in text:
        return "billing"

    if "crash" in text or "error" in text or "bug" in text:
        return "technical"

    if "cancel" in text:
        return "delivery"

    if "delivery" in text or "late" in text or "order" in text or "where" in text:
        return "delivery"

    return "general"

def get_priority(text):
    text = text.lower()

    if (
        "urgent" in text or
        "immediately" in text or
        "asap" in text or
        "charged" in text or
        "payment" in text
    ):
        return "high"

    return "low"


def run():
    env = SmartOpsEnv()
    obs = env.reset()
    total_reward = 0

    print("Starting SmartOps Agent\n")

    for step in range(10):

        if obs.category is None:
            category = classify_ticket(obs.user_message)
            action = Action(action_type="classify", content=category)

        elif obs.priority is None:
            priority = get_priority(obs.user_message)
            action = Action(action_type="prioritize", content=priority)

        elif len(obs.history) < 1:
            action = Action(
                action_type="respond",
                content="We understand your issue and refund will be processed if applicable."
            )

        else:
            action = Action(action_type="resolve", content="done")

        obs, reward, done, _ = env.step(action)
        total_reward += reward

        print(f"--- Step {step+1} ---")
        print("Ticket:", obs.user_message)
        print("Action:", action)
        print("Reward:", reward)
        print()

        if done:
            print("Task Completed")
            break

    print("Final Score:", total_reward)


if __name__ == "__main__":
    run()