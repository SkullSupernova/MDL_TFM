import matplotlib
matplotlib.use("Agg")  # backend headless para que las gráficas se generen sin display en tests

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.models import CHEXPERT_PATHOLOGY_COLS


@pytest.fixture
def synthetic_df(tmp_path):
    rows = []
    for i in range(4):
        img_array = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        img = Image.fromarray(img_array)
        p = tmp_path / f"img_{i}.jpg"
        img.save(str(p))
        row = {"Ruta_Absoluta": str(p)}
        for col in CHEXPERT_PATHOLOGY_COLS:
            row[col] = float(i % 2)
        rows.append(row)
    return pd.DataFrame(rows)
