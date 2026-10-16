#!/usr/bin/env python3
"""GPC-v2 LTCC 原子状态更新器；只更新状态 JSON，不改变实验参数。"""

import argparse
import json
import os
import socket
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--arm", default="未开始")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--checkpoint", default="无")
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--tmux-session", default="")
    parser.add_argument("--message", default="")
    args = parser.parse_args()

    payload = {
        "run_time": args.run_time,
        "主机": socket.gethostname(),
        "pid": args.pid or None,
        "tmux_session": args.tmux_session or None,
        "当前阶段": args.phase,
        "状态": args.state,
        "seed": args.seed,
        "arm": args.arm,
        "epoch": args.epoch,
        "最近checkpoint": args.checkpoint,
        "最后心跳时间": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "输出目录": args.output_root,
        "说明": args.message,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.path)), exist_ok=True)
    temporary = args.path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, args.path)


if __name__ == "__main__":
    main()
