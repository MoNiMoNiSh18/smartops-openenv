from fastapi import FastAPI
from env.environment import SmartOpsEnv
from env.models import Action

app = FastAPI()

env = SmartOpsEnv()


@app.post("/reset")
def reset():
    obs = env.reset()
    return {
        "observation": obs.dict(),
        "done": False
    }


@app.post("/step")
def step(action: dict):
    act = Action(**action)
    obs, reward, done, info = env.step(act)

    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info
    }


@app.get("/state")
def state():
    obs = env.state()
    return {
        "observation": obs.dict()
    }