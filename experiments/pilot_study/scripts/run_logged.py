from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

from common import RESULTS, ROOT, rel


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(description="Run one pilot script and capture stdout/stderr and exit code under results/logs")
    p.add_argument("--name", required=True, help="Short stage name used for the log filename")
    p.add_argument("command", nargs=argparse.REMAINDER, help="Script path followed by its arguments; place after --")
    args = p.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        p.error("Provide a script after --, e.g. -- scripts/run_baseline.py --scope sampled")
    script = Path(command[0])
    if not script.is_absolute():
        script = ROOT / script
    if script.suffix.lower() == ".py":
        command = [sys.executable, str(script), *command[1:]]
    log_path = RESULTS / "logs" / f"{args.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.datetime.now().astimezone().isoformat()
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    body = (process.stdout or "") + ("\n[stderr]\n" + process.stderr if process.stderr else "")
    log_path.write_text(f"started={started}\ncommand={' '.join(command)}\nexit_code={process.returncode}\n\n{body}", encoding="utf-8")
    if process.stdout:
        print(process.stdout, end="")
    if process.stderr:
        print(process.stderr, file=sys.stderr, end="")
    print(f"log={rel(log_path)} exit_code={process.returncode}")
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
