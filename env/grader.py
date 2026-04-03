def grade_classification(pred, true):
    return 1.0 if pred == true else 0.0

def grade_priority(pred, true):
    return 1.0 if pred == true else 0.0

def grade_response(response):
    response = response.lower()
    if "refund" in response:
        return 1.0
    elif "sorry" in response:
        return 0.5
    return 0.0