from .grader import *

def easy_task(env):
    return (
        grade_classification(env.predicted_category, env.current_ticket["true_category"]) +
        grade_priority(env.predicted_priority, env.current_ticket["true_priority"])
    ) / 2


def medium_task(response):
    return grade_response(response)


def hard_task(env, response_score):
    return (
        easy_task(env) * 0.5 +
        response_score * 0.5
    )