import pytest
import torch
from torchvision import transforms

from src.models import (
    CheXpertDataset,
    build_model,
    get_grad_cam_layer,
    CHEXPERT_PATHOLOGY_COLS,
    SUPPORTED_MODELS,
)


# =========================================================
# CHEXPERT_PATHOLOGY_COLS
# =========================================================

def test_pathology_count_es_trece():
    assert len(CHEXPERT_PATHOLOGY_COLS) == 13


def test_pathology_pleural_other_excluida():
    assert "Pleural Other" not in CHEXPERT_PATHOLOGY_COLS


def test_pathology_no_finding_incluida():
    assert "No Finding" in CHEXPERT_PATHOLOGY_COLS


# =========================================================
# build_model — happy path
# =========================================================

@pytest.mark.parametrize("model_name", ["densenet121", "resnet50", "efficientnet_b0"])
def test_build_model_shape_correcta(model_name):
    model = build_model(model_name=model_name, num_classes=13, pretrained=False)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 13)


def test_build_model_num_classes_personalizado():
    model = build_model(model_name="densenet121", num_classes=5, pretrained=False)
    out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 5)


# =========================================================
# build_model — edge cases y errores
# =========================================================

def test_build_model_nombre_invalido_lanza_valueerror():
    with pytest.raises(ValueError, match="no soportado"):
        build_model(model_name="vgg16")


def test_build_model_sin_pretrained_no_descarga():
    # pretrained=False no lanza error aunque no haya red
    model = build_model(model_name="resnet50", num_classes=13, pretrained=False)
    assert model is not None


# =========================================================
# CheXpertDataset — happy path
# =========================================================

def test_dataset_len_correcto(synthetic_df):
    ds = CheXpertDataset(synthetic_df)
    assert len(ds) == 4


def test_dataset_item_shapes_correctas(synthetic_df):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    ds = CheXpertDataset(synthetic_df, transform=tf)
    img, label = ds[0]
    assert img.shape == (3, 224, 224)
    assert label.shape == (13,)


def test_dataset_etiquetas_cols_personalizada(synthetic_df):
    cols = CHEXPERT_PATHOLOGY_COLS[:3]
    ds = CheXpertDataset(synthetic_df, etiquetas_cols=cols)
    _, label = ds[0]
    assert label.shape == (3,)


# =========================================================
# CheXpertDataset — edge cases
# =========================================================

def test_dataset_sin_transform_devuelve_pil(synthetic_df):
    from PIL import Image as PILImage
    ds = CheXpertDataset(synthetic_df, transform=None)
    img, _ = ds[0]
    assert isinstance(img, PILImage.Image)


def test_dataset_label_dtype_float32(synthetic_df):
    tf = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    ds = CheXpertDataset(synthetic_df, transform=tf)
    _, label = ds[0]
    assert label.dtype == torch.float32


# =========================================================
# get_grad_cam_layer — happy path
# =========================================================

@pytest.mark.parametrize("model_name", ["densenet121", "resnet50", "efficientnet_b0"])
def test_grad_cam_layer_devuelve_lista_de_uno(model_name):
    model = build_model(model_name, pretrained=False)
    layer = get_grad_cam_layer(model, model_name)
    assert isinstance(layer, list)
    assert len(layer) == 1


# =========================================================
# get_grad_cam_layer — errores
# =========================================================

def test_grad_cam_layer_nombre_invalido_lanza_valueerror():
    model = build_model("densenet121", pretrained=False)
    with pytest.raises(ValueError):
        get_grad_cam_layer(model, "vgg16")
