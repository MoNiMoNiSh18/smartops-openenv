from fastapi import FastAPI
from pydantic import BaseModel
from env.environment import SmartOpsEnv
from env.models import Action

app = FastAPI()

env = SmartOpsEnv()


class StepRequest(BaseModel):
    action_type: str
    content: str


@app.post("/reset")
def reset():
    obs = env.reset()
    return {"observation": obs.dict()}


@app.post("/step")
def step(req: StepRequest):
    action = Action(action_type=req.action_type, content=req.content)
    obs, reward, done, _ = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()