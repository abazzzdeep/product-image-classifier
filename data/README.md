# Dataset

This project needs product images organized into three folders:

```
data/raw/
├── Apparel/
├── Electronics/
└── Home/
```

## Suggested sources (public, free)

- **Apparel** → [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) — Myntra-sourced (Indian e-commerce), ships with a `styles.csv` metadata file that includes a `masterCategory` column you can filter down to `Apparel`.
- **Electronics / Home** → [ecommerce_product_images_18K](https://www.kaggle.com/datasets/fatihkgg/ecommerce-product-images-18k) — ~18K images across 9 marketplace categories (Amazon/Walmart-sourced). After downloading, check the actual folder/category names and pull whichever map closest to Electronics and Home — they may not be named exactly that.

You don't need the full datasets — roughly 800–1,500 images per class is plenty for a small CNN trained on Colab.

## Downloading via the Kaggle API (inside Colab)

```python
from google.colab import files
files.upload()  # upload your kaggle.json (from kaggle.com/settings)

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d paramaggarwal/fashion-product-images-small
!unzip -q fashion-product-images-small.zip -d apparel_raw

!kaggle datasets download -d fatihkgg/ecommerce-product-images-18k
!unzip -q ecommerce-product-images-18k.zip -d ecom_raw
```

Then sort images into `data/raw/Apparel`, `data/raw/Electronics`, `data/raw/Home` based on each source's labels/folders.

**Never commit `kaggle.json`** — it's already excluded in `.gitignore`.
