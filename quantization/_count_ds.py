import os
root = "/home/tomi/code/dipl/datasets/mini_set/sub10k_open_images_v7"
names = ["Person", "Car", "Truck", "Bus", "Motorcycle", "Bicycle"]
exts = (".jpg", ".jpeg", ".png")
hdr = f"{'split':<8}{'images':>8}" + "".join(f"{n:>11}" for n in names) + f"{'tot.box':>10}"
print(hdr)
gt_img = gt_box = 0
perc_tot = [0] * 6
for sp in ["train", "val", "test"]:
    imgd = os.path.join(root, "images", sp)
    lbd = os.path.join(root, "labels", sp)
    nimg = len([f for f in os.listdir(imgd) if f.lower().endswith(exts)]) if os.path.isdir(imgd) else 0
    perc = [0] * 6
    if os.path.isdir(lbd):
        for f in os.listdir(lbd):
            if not f.endswith(".txt"):
                continue
            for ln in open(os.path.join(lbd, f)).read().splitlines():
                p = ln.split()
                if len(p) == 5:
                    c = int(p[0])
                    if 0 <= c < 6:
                        perc[c] += 1
    tb = sum(perc)
    print(f"{sp:<8}{nimg:>8}" + "".join(f"{perc[i]:>11}" for i in range(6)) + f"{tb:>10}")
    gt_img += nimg; gt_box += tb
    for i in range(6):
        perc_tot[i] += perc[i]
print(f"{'UKUPNO':<8}{gt_img:>8}" + "".join(f"{perc_tot[i]:>11}" for i in range(6)) + f"{gt_box:>10}")
