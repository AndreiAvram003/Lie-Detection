import random

def mock_process_video(file_path: str):
    return {
        "truth": random.choice([True, False]),
        "confidence": round(random.uniform(60.0, 99.9), 2)
    }
