import csv
import io
import os

Q = '/home/tomi/code/dipl/quantization'

M = {
    'schoolcnn': ('map_macro', 'PTQ/schoolcnn/schoolcnn_ptq_report.csv',
                  'QAT/results2/schoolcnn/schoolcnn_report.csv', 'INT8-PT-static'),
    'distilbert': ('acc', 'PTQ/distilbert/distilbert_ptq_report.csv',
                   'QAT/results2/distilbert/distilbert_report.csv', 'INT8-ORT-dynamic'),
    'deeplabv3': ('miou', 'PTQ/deeplabv3/deeplabv3_ptq_report.csv',
                  'QAT/results2/deeplabv3/deeplabv3_report.csv', 'INT8-PT-static'),
    'yolo26n': ('map', 'PTQ/yolo26n/yolo26n_ptq_report.csv',
                'QAT/results2/yolo26n/yolo26n_report.csv', 'INT8-static'),
    'yolo26l': ('map', 'PTQ/yolo26l/yolo26l_ptq_report.csv',
                'QAT/results2/yolo26l/yolo26l_report.csv', 'INT8-static'),
}
CPU_BE1 = {'INT8-PT-static': 'PT-x86', 'INT8-ORT-dynamic': 'ORT', 'INT8-static': 'OpenVINO'}


def load(p):
    return list(csv.DictReader(io.open(os.path.join(Q, p), encoding='utf-8')))


def num(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError):
        return None


def pick1(rows, fmt, met):
    for r in rows:
        if r.get('format') == fmt:
            return num(r, met)
    return None


def pick2(rows, proc, fmt, met):
    for r in rows:
        if r.get('postupak') == proc and r.get('format') == fmt:
            return num(r, met)
    return None


def cell(v, base):
    if v is None:
        return '—'
    if base is None:
        return f'{v:.4f}'
    return f'{v:.4f} ({v - base:+.4f})'


out = []
out.append('| model | metrika | red | PTQ run1 | PTQ run2 | QAT run2 |')
out.append('|---|---|---|---|---|---|')

for name, (met, p1, p2, cpu1) in M.items():
    r1, r2 = load(p1), load(p2)
    b1 = pick1(r1, 'FP32', met)
    b2p = pick2(r2, 'PTQ', 'FP32', met)
    b2q = pick2(r2, 'QAT', 'FP32', met)
    orig = pick2(r2, 'ORIG', 'FP32', met)

    out.append(f'| **{name}** | {met} | FP32 osnovica | {b1:.4f} | {b2p:.4f} | {b2q:.4f} |')
    out.append(f'| | | INT8 CPU | {cell(pick1(r1, cpu1, met), b1)} `{CPU_BE1[cpu1]}` '
               f'| {cell(pick2(r2, "PTQ", "INT8-OV", met), b2p)} `OV` '
               f'| {cell(pick2(r2, "QAT", "INT8-OV", met), b2q)} `OV` |')
    out.append(f'| | | INT8 TRT | {cell(pick1(r1, "INT8-TRT", met), b1)} '
               f'| {cell(pick2(r2, "PTQ", "INT8-TRT", met), b2p)} '
               f'| {cell(pick2(r2, "QAT", "INT8-TRT", met), b2q)} |')
    out.append(f'| | | FP16 TRT | {cell(pick1(r1, "FP16-TRT", met), b1)} '
               f'| {cell(pick2(r2, "PTQ", "FP16-TRT", met), b2p)} '
               f'| {cell(pick2(r2, "QAT", "FP16-TRT", met), b2q)} |')
    out.append(f'| | | *(ORIG run2)* | | {orig:.4f} | {orig:.4f} |')

t = '\n'.join(out)
io.open('/home/tomi/code/dipl/quantization/QAT/results2/run1_vs_run2.md', 'w',
        encoding='utf-8').write(t + '\n')
print(t)
