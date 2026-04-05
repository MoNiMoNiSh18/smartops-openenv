import os
from typing import List, Optional
from openai import OpenAI

from env.environment import SmartOpsEnv
from env.models import Action

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

TASK_NAME = "smartops-ticket-resolution"
BENCHMARK = "smartops-env"

MAX_STEPS = 6
SUCCESS_SCORE_THRESHOLD = 0.3


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def classify_ticket(text):
    text = text.lower()

    if "refund" in text or "damaged" in text or "poor" in text or "wrong" in text:
        return "refund"

    if "delivery" in text or "late" in text or "order" in text or "where" in text:
        return "delivery"

    if "payment" in text or "charged" in text or "invoice" in text or "purchase" in text:
        return "billing"

    if "crash" in text or "error" in text or "bug" in text:
        return "technical"

    if "cancel" in text:
        return "delivery"

    return "general"


def prioritize_ticket(text):
    text = text.lower()

    if "urgent" in text or "immediately" in text:
        return "high"

    if "refund" in text or "charged" in text or "crash" in text:
        return "high"

    if "late" in text or "order" in text:
        return "medium"

    return "low"


def run():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = SmartOpsEnv()
    obs = env.reset()

    rewards = []
    steps_taken = 0

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):
            text = obs.user_message

            if obs.category is None:
                category = classify_ticket(text)
                action = Action(action_type="classify", content=category)

            elif obs.priority is None:
                priority = prioritize_ticket(text)
                action = Action(action_type="prioritize", content=priority)

            elif len(obs.history) < 2:
                action = Action(
                    action_type="respond",
                    content="We understand your issue and are working on resolving it."
                )

            else:
                action = Action(action_type="resolve", content="done")

            obs, reward, done, _ = env.step(action)

            rewards.append(reward)
            steps_taken = step

            log_step(step, str(action), reward, done, None)

            if done:
                break

        score = sum(rewards) / (len(rewards) * 1.0) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        log_step(steps_taken + 1, "error", 0.0, True, str(e))
        success = False
        score = 0.0

    log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    run()