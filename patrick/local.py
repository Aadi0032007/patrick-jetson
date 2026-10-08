from .ports import Response


class LocalProvider:
    """Offline starter responses. No invented robot telemetry or movement execution."""
    def __init__(self):
        self.responses = []

    def submit(self, turn_id, text):
        if "battery" in text.lower():
            answer = "Battery telemetry is not connected yet."
        else:
            answer = "I heard: " + text + ". My cloud conversation provider is not connected yet."
        self.responses.append(Response(turn_id, text=answer, final=True))

    def cancel(self, turn_id):
        self.responses = [r for r in self.responses if r.turn_id != turn_id]

    def poll(self):
        responses, self.responses = self.responses, []
        return responses


class SimRobot:
    def __init__(self):
        self.moving = False
        self.stop_count = 0
        self.cancel_count = 0

    def stop_movement(self):
        self.moving = False
        self.stop_count += 1

    def cancel_actions(self):
        self.cancel_count += 1


class ConsolePlayback:
    def __init__(self):
        self.busy = False

    def enqueue(self, response):
        if response.text:
            print("Patrick:", response.text)

    def stop(self):
        self.busy = False

    def beep(self):
        print("[listening acknowledgement]")
