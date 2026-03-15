import time

class Timer:



    def start(self):
        self.start_time = time.time()

    def stop(self):
        elapsed = time.time() - self.start_time
        minutes = elapsed // 60
        seconds = elapsed % 60
        return f"{int(minutes)} min {int(seconds)} sec"





