import time
import click
from piper_sdk import *

@click.command()
@click.option('--dual','-d',required=False, default=False)
def __main__(dual):
    pipers = []
    try:
        piper = C_PiperInterface_V2("can0")
        piper.ConnectPort()
        while not piper.EnablePiper():
            time.sleep(0.01)
        pipers.append(piper)

        piper.MotionCtrl_2(0x01, 0x01, 20, 0x00)
        piper.JointCtrl(int(0/0.001), int(50/0.001), int(-50/0.001), 0, int(50/0.001), 0)
        print(dual)

        if dual:
            piper2 = C_PiperInterface_V2("can1")
            piper2.ConnectPort()
            while not piper2.EnablePiper():
                time.sleep(0.01)
            pipers.append(piper2)

            piper2.MotionCtrl_2(0x01, 0x01, 20, 0x00)
            piper2.JointCtrl(int(-90/0.001), int(50/0.001), int(-50/0.001), 0, int(50/0.001), 0)

        time.sleep(1.0)  # wait for joints to arrive
        print("Done. Press Ctrl+C to exit.")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        for p in pipers:
            try:
                # p.DisableArm()
                print("error")
            except Exception:
                pass
            time.sleep(0.05)
            try:
                p.DisconnectPort()
            except Exception:
                pass
        print("Arms disabled, CAN ports closed.")

if __name__ == '__main__':
    __main__()