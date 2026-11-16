# Sistemi i rekomandimit të filmave

Case study 9, Artificial Intelligence (2026/2027). Projekti përdor MovieLens Latest Small për të gjetur filma të ngjashëm sipas zhanrit (TF-IDF + kosinus). Ndërfaqja Flask ofron edhe rekomandime për përdoruesit e MovieLens me modelin SVD.

## Materialet e dorëzimit

- [Notebook-u me fazat CRISP-DM](case-study/Rekomandimi.ipynb) · në Google Colab zgjidh **File → Upload notebook**, pastaj **Runtime → Run all**. Kur dataset-i lokal mungon, notebook-u shkarkon MovieLens nga GroupLens.
- [Raporti me 5 faqe](case-study/Raporti.pdf)
- [Prezantimi me shënime dhe kohëzgjatje 8 minuta](case-study/Prezantimi.pptx)
- [Diagrami i rrjedhës së të dhënave](case-study/diagrams/dfd_level1.png) dhe [burimi i editueshëm në draw.io](case-study/diagrams/dfd_level1.drawio)

Provo `rekomando_filma("Toy Story (1995)", n=5)` në notebook. Metrika HitRate@10 krahason ngjashmërinë e zhanreve me popullaritetin në 608 raste. Për SVD, aplikacioni mat RMSE të notave të fshehura.

## Nisja e aplikacionit

Nga dosja kryesore e projektit, me Python 3.12:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Hape [aplikacionin lokal në portën 5001](http://127.0.0.1:5001). Provo filmin `Toy Story (1995)` ose përdoruesin `1`.

```sh
python -m pytest -q
python recommender.py
python make_diagrams.py
```

Kushtet dhe citimi i dataset-it gjenden te [dokumentimi i MovieLens](data/ml-latest-small/README.txt).
