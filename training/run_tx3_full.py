import json
from pathlib import Path
import subprocess
import shutil

SUMMARY_PATH = Path("weight_v2/summary_delay120_v2.json")
WEIGHT_PATH = Path("weight_v2/delay_120_weight_v2.json")

# 1. 清空 summary/weights，让脚本重新跑 CA_1…TX_3，不会跳过
for path in (SUMMARY_PATH, WEIGHT_PATH):
    if path.exists():
        path.write_text("[]")

# 2. 只跑一个“小” store（TX_3）
cmd = [
    "python",
    "train_lgbm_baseline.py",
    "--stores",
    "TX_3",
]
subprocess.run(cmd, check=True)

# 3. 运行 visualize_and_blend 生成 base/alt/bld submissions
visualize_cmd = [
    "python",
    "visualize_and_blend.py",
    "--summary",
    str(SUMMARY_PATH),
    "--weights",
    str(WEIGHT_PATH),
    "--base",
    "future_finaldata/submission_with_val.csv",
    "--alt",
    "future_finaldata/submission_with_val_cmodel.csv",
    "--out",
    "blended/submission_with_val_blended.csv",
    "--fig",
    "blended/wrmsse_scatter.png",
]
subprocess.run(visualize_cmd, check=True)
