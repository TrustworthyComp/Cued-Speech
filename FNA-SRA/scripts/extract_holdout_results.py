"""
从 holdout 训练日志中提取最佳 WER/CER/BLEU4。

用法:
    python scripts/extract_holdout_results.py logs/holdout_03_13.log
"""
import re, sys, os

log_file = sys.argv[1] if len(sys.argv) > 1 else 'logs/holdout.log'

if not os.path.exists(log_file):
    print(f'[ERROR] 日志文件不存在: {log_file}')
    sys.exit(1)

content = open(log_file).read()

# holdout_test 作为 validation 集运行 → 指标命名空间是 val/
cer_m = re.findall(r"val/cer['\"]?\s*[:=]\s*([\d.]+)", content)
wer_m = re.findall(r"val/wer['\"]?\s*[:=]\s*([\d.]+)", content)

print(f"日志文件 : {log_file}")
print(f"Best CER : {float(cer_m[-1]):.2f}%"  if cer_m else "CER : N/A（训练未到验证周期）")
print(f"Best WER : {float(wer_m[-1]):.2f}%"  if wer_m else "WER : N/A")

if cer_m:
    print(f"\n所有 val/cer 记录（共 {len(cer_m)} 次验证，每 2 epoch 一次）:")
    for i, (c, w) in enumerate(zip(cer_m, wer_m if wer_m else ['-']*len(cer_m))):
        print(f"  epoch {(i+1)*2:>4}: CER={float(c):.2f}%  WER={float(w):.2f}%" if w != '-' else f"  epoch {(i+1)*2:>4}: CER={float(c):.2f}%")
