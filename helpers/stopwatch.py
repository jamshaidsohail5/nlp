import time


class StopWatch():

    def __init__(self, msg, freq, total='-'):
        self.msg = msg
        self.freq = freq
        self.total = total

    def start(self):
        self.t0 = time.time()

    def run(self, cnt):
        if cnt % self.freq == 0 or cnt == self.total:
            elapsed = time.time() - self.t0
            hour = elapsed // 3600
            minute = (elapsed - hour * 3600) // 60
            sec = elapsed % 60

            if self.total != '-':
                share = round(cnt / self.total, 2)
            else:
                share = self.total

            print(
                '\r{message:s} {cnt:.0f} ({share} %), elapsed time: {hour:.0f} h {minute:.0f} m {sec:.0f} s     '.format(  # noqa
                message=self.msg,
                cnt=cnt,
                share=share,
                hour=hour,
                minute=minute,
                sec=sec
                ),
            end='',
            flush=True
            )
