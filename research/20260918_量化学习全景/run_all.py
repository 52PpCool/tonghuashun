# -*- coding: utf-8 -*-
"""
一键复跑全部课程（自检脚本）。
用法:
    python run_all.py            # 复跑 01-32 全部课程（17课跑解析打分阶段）
    python run_all.py --quick    # 只复跑第一批 01-17
结果写入 run_all_report.txt；任何一课失败会保留 stderr 到 failed_*.txt。
"""
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(BASE, ".venv", "bin", "python")

COURSE_SCRIPTS = {
    "17_iFinD真实数据管道": "parse_and_screen.py",   # 17课的取数已留痕，复跑解析打分阶段
    "29_PEAD财报漂移": "parse_and_study.py",         # 29课复跑事件研究阶段（取数已留痕）
}


def main():
    quick = "--quick" in sys.argv
    courses = sorted(
        d for d in os.listdir(BASE)
        if os.path.isdir(os.path.join(BASE, d))
        and d[:2].isdigit() and not d.startswith(("data", "."))
    )
    if quick:
        courses = [c for c in courses if c[:2] <= "17"]

    report = ["课程 | 退出码 | 耗时(秒)", "-" * 40]
    failed = 0
    for course in courses:
        script = COURSE_SCRIPTS.get(course, "main.py")
        start = time.time()
        proc = subprocess.run(
            [PY, script], cwd=os.path.join(BASE, course),
            capture_output=True, text=True, timeout=300,
        )
        elapsed = round(time.time() - start, 1)
        status = "OK" if proc.returncode == 0 else "FAIL"
        if proc.returncode != 0:
            failed += 1
            err_path = os.path.join(BASE, f"failed_{proc.returncode}_{course}.txt")
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(proc.stderr)
        report.append(f"{course} | {status} | {elapsed}")
        print(f"[{status}] {course} ({elapsed}s)")

    summary = "\n".join(report) + f"\n\n合计 {len(courses)} 课，失败 {failed} 课\n"
    with open(os.path.join(BASE, "run_all_report.txt"), "w", encoding="utf-8") as f:
        f.write(summary)
    print(summary)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
