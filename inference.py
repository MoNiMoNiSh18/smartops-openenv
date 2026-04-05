import requests

BASE_URL = "http://localhost:7860"

def run_agent():
    obs = requests.post(f"{BASE_URL}/reset").json()

    total_reward = 0

    for step in range(10):
        text = obs["user_message"].lower()

        if obs["category"] is None:
            if "refund" in text or "damaged" in text or "poor" in text or "wrong" in text:
                action = {"action_type": "classify", "content": "refund"}
            elif "delivery" in text or "late" in text or "order" in text or "where" in text or "cancel" in text:
                action = {"action_type": "classify", "content": "delivery"}
            elif "payment" in text or "charged" in text or "invoice" in text:
                action = {"action_type": "classify", "content": "billing"}
            elif "crash" in text or "error" in text:
                action = {"action_type": "classify", "content": "technical"}
            else:
                action = {"action_type": "classify", "content": "general"}

        elif obs["priority"] is None:
            if "urgent" in text or "immediately" in text:
                action = {"action_type": "prioritize", "content": "high"}
            elif "refund" in text or "charged" in text:
                action = {"action_type": "prioritize", "content": "high"}
            elif "late" in text or "delivery" in text:
                action = {"action_type": "prioritize", "content": "medium"}
            else:
                action = {"action_type": "prioritize", "content": "low"}

        else:
            action = {"action_type": "resolve", "content": "done"}

        response = requests.post(f"{BASE_URL}/step", json=action).json()

        obs = response["observation"]
        reward = response["reward"]
        done = response["done"]

        total_reward += reward

        if done:
            break

    print("Final Score:", total_reward)


if __name__ == "__main__":
    run_agent()