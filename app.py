from fastapi import FastAPI
from pydantic import BaseModel
from env.environment import SmartOpsEnv
from env.models import Action

app = FastAPI()

env = SmartOpsEnv()
obs = env.reset()


class StepInput(BaseModel):
    action_type: str
    content: str


@app.post("/reset")
def reset():
    global obs
    obs = env.reset()
    return obs.dict()


@app.post("/step")
def step(input: StepInput):
    global obs

    action = Action(
        action_type=input.action_type,
        content=input.content
    )

    obs, reward, done, _ = env.step(action)

    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done
    }


@app.get("/")
def home():
    return {"message": "SmartOps OpenEnv API running "}