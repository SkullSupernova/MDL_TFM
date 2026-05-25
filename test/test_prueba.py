import pandas
import pytest

from src.utils import load_data

print(pytest.__version__)

# scope -> function, class, module, session
@pytest.fixture(scope="session")
def test_x():
    pass

def test_carga_dataset():
    df = pandas.read_csv("../Modulo 2/Credit_score_v0/credit_score_dataset.csv")
    assert len(df) > 0, "El dataset no se ha cargado correctamente o está vacío."

def test_carga_dataset_2():
    df = load_data("../Modulo 2/Credit_score_v0/credit_score_dataset.csv")
    assert len(df) > 0, "El dataset no se ha cargado correctamente o está vacío."

def test_dataset_etiqueta():
    df = load_data("../Modulo 2/Credit_score_v0/credit_score_dataset.csv")
    assert "risk" in df.columns, "La columna 'risk' no se encuentra en el dataset."