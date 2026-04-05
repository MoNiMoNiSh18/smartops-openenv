---
title: SmartOps OpenEnv
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---


SmartOps OpenEnv: AI Ticket Resolution Environment

Overview:
SmartOps is a real-world OpenEnv environment simulating customer support ticket resolution workflows. It enables AI agents to learn structured decision-making using the standard step(), reset(), and state() interface.

Problem Statement
Customer support teams handle thousands of tickets daily involving:

Issue classification
Priority assignment
Response generation
Resolution

This environment models that workflow for training and evaluating AI agents.

Environment Design:
*Observation Space
Each state contains:

ticket_id
user_message
category (optional)
priority (optional)
history
time_elapsed

*Action Space
Agents can perform:

classify → assign category
prioritize → assign priority
respond → generate reply
resolve → close ticket

*Reward Function
Correct classification: +0.3
Correct prioritization: +0.3
Response generation: +0.2
Successful resolution: +0.5
Wrong actions: penalties
Repeated actions: penalty
Step penalty: -0.05

Tasks
Difficulty	Task Description
Easy	    Classify ticket correctly
Medium	    Classify + prioritize
Hard	    Full pipeline: classify → prioritize → respond → resolve

Baseline Agent
A rule-based agent is provided:
python -m baseline.run_agent

Docker Support
Run anywhere:
docker build -t smartops .
docker run smartops

Dataset
Stored in tickets.json:

Realistic customer queries
Multiple categories (billing, delivery, refund, technical)
Varied priorities

Deployment
Live on Hugging Face Spaces (Docker-based environment)

OpenEnv Compliance
step() / reset() / state()
Typed Pydantic models
Deterministic graders
Multi-task evaluation

Future Improvements
Add multi-turn conversations
Integrate real LLM agent
Expand dataset
Add noisy/ambiguous tickets