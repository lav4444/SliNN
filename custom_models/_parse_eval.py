import os, re
base = "/home/tomi/code/dipl/custom_models"
students = ["student_0_5_m", "student_1_m", "student_2_m", "student_yolo26n"]
kinds = ["pure_KD", "KD_featlogit"]

def parse(path):
    if not os.path.exists(path):
        return None
    txt = open(path).read()
    pm = re.search(r"\(([\d,]+)\s*params", txt)
    params = pm.group(1).replace(",", "") if pm else "?"
    out = {}
    for split in ["TRAIN", "VAL", "TEST"]:
        m = re.search(split + r".*?mAP@50:95\s*=\s*([\d.]+)", txt, re.S)
        out[split] = m.group(1) if m else "?"
    return params, out

print(f"{'student':<16}{'kind':<14}{'params':>10}{'VAL':>9}{'TEST':>9}")
for s in students:
    for k in kinds:
        r = parse(os.path.join(base, s, k, "eval_result.txt"))
        if r:
            p, m = r
            print(f"{s:<16}{k:<14}{p:>10}{m['VAL']:>9}{m['TEST']:>9}")
