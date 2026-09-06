# legacy

Arhiva, IZVAN puta izvodjenja. Zivi proizvod je `slinn/`.

- `morphology/` — izvorni detekcijski pipeline. Dokazane prune/grow/dead mehanike su
  preseljene u `slinn/morph.py`, decode u `slinn/plugins/detection/`.
  Teacher cache (`tmp/`, 12 GB) obrisan kao regenerabilan; `models/` (rezultati ranijih
  kompresija) ZADRZAN.
- `arch_agnostic/` — agnosticna jezgra prije mergea; `tmp/` zadrzan.

Ne uvoziti odavde. Jedini preostali korisnici su `slinn/helper/_parity60.py`
(usporeduje stari i novi engine) i `baseline_models/*/_verify.py`.
