"""Run long Colab commands as background jobs, so a browser disconnect does not stop them.

    launch('P1_unet_2d', 'python -m src.segmentation.train ... --run P1_unet_2d')  # starts it, or re-attaches
    follow('P1_unet_2d')   # show progress again after reconnecting
    stop('P1_unet_2d')     # end the job itself (interrupting a cell only stops watching)

A job survives the browser disconnecting, but not a runtime reset; the training commands resume from their last
checkpoint, so after a reset launch() the same job again.
"""

import os
import re
import signal
import subprocess
import time

LOGS = '/content/logs'
# Messages from libraries Colab preloads; they say nothing about our runs.
NOISE = re.compile(r'absl|cpu_feature_guard|cudart_stub|TensorFlow|nvtiff|nvjpeg2k|nvImageCodec|nvTIFF|nvJPEG2000')


def _pids(name):
    # Every job's shell command ends with a "# job=<name>" marker, which pgrep finds in its command line.
    found = subprocess.run(['pgrep', '-f', '--', f'# job={name}$'], capture_output=True, text=True).stdout
    return [int(pid) for pid in found.split()]


def running(name):
    return bool(_pids(name))


def log_path(name, logs=LOGS):
    return os.path.join(logs, f'{name}.log')


def launch(name, command, cwd='/content/Panvision', logs=LOGS, poll=20):
    if running(name):
        print(f'{name} is already running; showing its log')
    else:
        os.makedirs(logs, exist_ok=True)
        log = log_path(name, logs)
        shell = f'cd {cwd} && ({command}) >> {log} 2>&1; echo "EXIT $?" >> {log} # job={name}'
        subprocess.Popen(['bash', '-c', shell], start_new_session=True)
        time.sleep(2)
    return follow(name, logs, poll)


def follow(name, logs=LOGS, poll=20):
    """Print the job's log as it grows until the job ends; returns True if it exited cleanly."""
    log, position = log_path(name, logs), 0
    while True:
        alive = running(name)
        if os.path.exists(log):
            with open(log) as handle:
                handle.seek(position)
                text, position = handle.read(), handle.tell()
            for line in text.splitlines():
                if not NOISE.search(line):
                    print(line, flush=True)
        if not alive:
            break
        time.sleep(poll)
    lines = open(log).read().strip().splitlines() if os.path.exists(log) else []
    last = lines[-1] if lines else 'no log'
    if last == 'EXIT 0':
        print(f'{name} finished')
        return True
    hint = ('it ran out of RAM' if last in ('EXIT 137', 'EXIT -9') or 'Killed' in ''.join(lines[-3:])
            else 'see the error above')
    print(f'{name} stopped ({last}): {hint}. Run the same cell again to resume from the last checkpoint.')
    return False


def stop(name):
    for pid in _pids(name):
        try:
            os.killpg(pid, signal.SIGTERM)  # the job's whole process group: the shell and the command under it
        except ProcessLookupError:
            pass
