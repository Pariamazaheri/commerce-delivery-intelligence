# Publish the portfolio project

Canonical repository: https://github.com/Pariamazaheri/commerce-delivery-intelligence

## GitHub

Create a public repository named `commerce-delivery-intelligence` in your account,
without auto-generating a README. From this project's directory:

```bash
git add .
git commit -m "Build reproducible delivery reliability analytics"
git remote add origin https://github.com/Pariamazaheri/commerce-delivery-intelligence.git
git push -u origin main
```

Commit identity and account authentication come from your Git configuration. The data,
virtual environment, secrets and caches are ignored. The executed notebook,
figures, measured reports, tests and 50-car Excel snapshot are included.

Suggested GitHub description:

> Order-level delivery reliability analytics with temporal evaluation, mutual
> information, PCA, uncertainty estimates and an interactive route explorer.

Suggested topics: `data-science`, `ecommerce`, `exploratory-data-analysis`,
`feature-engineering`, `scikit-learn`, `plotly`, `pca`.

Pin the repository on your profile. Lead recruiter conversations with the business
question, order-grain design, training-label cutoff and the observed failure of
MI selection to improve temporal generalization. Avoid claiming deployment,
business savings or causal impact that were not measured.

## Colab

Once published, use Colab's **Open notebook > GitHub** dialog and select the
notebook from your repository. Before its first analytical cell, add and run:

```python
import subprocess
import os

repo_url = "https://github.com/Pariamazaheri/commerce-delivery-intelligence.git"
subprocess.run(["git", "clone", repo_url, "/content/commerce-project"], check=True)
os.chdir("/content/commerce-project")
subprocess.run(["python", "-m", "pip", "install", ".[dev]"], check=True)
subprocess.run(["commerce-insights", "download"], check=True)
```

The notebook's root detection then
finds the package and data. If Kaggle download fails, upload its six required CSVs
into `/content/commerce-project/data/raw/`. Save a Colab copy in your Drive and
use **Share > Anyone with the link** if you want a publicly accessible copy.

You can then add the real GitHub and Colab links to a separate handoff message.
The actual `.ipynb` file is included in this delivery and remains directly usable.
