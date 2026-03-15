from time import sleep

from src.utils.timer import Timer
def test_timer():
    timer = Timer()
    timer.start()
    sleep(2)
    end = timer.stop()
    assert end == "0 min 2 sec"
