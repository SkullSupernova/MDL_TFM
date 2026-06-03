import pytest
import torch
from torchvision import transforms

from src.models import (
    CheXpertDataset,
    build_model,
    get_grad_cam_layer,
    load_checkpoint,
    get_active_pathology_cols,
    CHEXPERT_PATHOLOGY_COLS,
    CHEXPERT_COMPETITION_5,
    CLASS_CONFIGS,
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

@pytest.mark.parametrize(
    "model_name", ["densenet121", "vgg16", "resnet50", "efficientnet_b0", "convnext_tiny"]
)
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
        build_model(model_name="mobilenet_v2")


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

@pytest.mark.parametrize(
    "model_name", ["densenet121", "vgg16", "resnet50", "efficientnet_b0", "convnext_tiny"]
)
def test_grad_cam_layer_devuelve_lista_de_uno(model_name):
    model = build_model(model_name, pretrained=False)
    layer = get_grad_cam_layer(model, model_name)
    assert isinstance(layer, list)
    assert len(layer) == 1


def test_grad_cam_layer_vgg16_es_conv2d():
    import torch.nn as nn
    model = build_model("vgg16", pretrained=False)
    capa = get_grad_cam_layer(model, "vgg16")[0]
    assert isinstance(capa, nn.Conv2d)


# =========================================================
# get_grad_cam_layer — errores
# =========================================================

def test_grad_cam_layer_nombre_invalido_lanza_valueerror():
    model = build_model("densenet121", pretrained=False)
    with pytest.raises(ValueError):
        get_grad_cam_layer(model, "mobilenet_v2")


# =========================================================
# CLASS_CONFIGS / get_active_pathology_cols
# =========================================================

def test_class_config_full13_tiene_trece():
    assert len(get_active_pathology_cols("full13")) == 13


def test_class_config_nofracture12_excluye_fracture():
    cols = get_active_pathology_cols("nofracture12")
    assert len(cols) == 12
    assert "Fracture" not in cols


def test_class_config_min5pct9_excluye_cuatro_clases():
    cols = get_active_pathology_cols("min5pct9")
    assert len(cols) == 9
    for excluida in ["Enlarged Cardiomediastinum", "Lung Lesion", "Pneumonia", "Fracture"]:
        assert excluida not in cols


@pytest.mark.parametrize("config", ["full13", "nofracture12", "min5pct9"])
def test_class_config_conserva_las_cinco_de_chexpert(config):
    cols = get_active_pathology_cols(config)
    assert set(CHEXPERT_COMPETITION_5).issubset(cols)


def test_class_config_invalido_lanza_valueerror():
    with pytest.raises(ValueError, match="class_config"):
        get_active_pathology_cols("config_inexistente")


def test_class_configs_define_modo_anti_ruido():
    assert CLASS_CONFIGS["full13"]["anti_ruido"] == "ninguno"
    assert CLASS_CONFIGS["nofracture12"]["anti_ruido"] == "orfanos"
    assert CLASS_CONFIGS["min5pct9"]["anti_ruido"] == "sin_positivos"


# =========================================================
# load_checkpoint — round-trip con backbone nuevo
# =========================================================

def test_load_checkpoint_round_trip_convnext_tiny(tmp_path):
    import torch as _torch
    cfg = {"model": {"name": "convnext_tiny", "dropout": 0.5, "hidden_units": 256}}
    model = build_model("convnext_tiny", num_classes=9, hidden_units=256, pretrained=False)
    ckpt = tmp_path / "convnext.pth"
    _torch.save(model.state_dict(), ckpt)

    cargado, num_classes = load_checkpoint(cfg, str(ckpt), _torch.device("cpu"))
    assert num_classes == 9
    out = cargado(_torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 9)


# =========================================================
# parse_checkpoint_filename
# =========================================================

@pytest.mark.parametrize("nombre,esperado", [
    ("mejor_modelo_densenet121_full13.pth", ("densenet121", "full13")),
    ("mejor_modelo_resnet50_nofracture12.pth", ("resnet50", "nofracture12")),
    # Backbone con guion bajo: el sufijo de config debe detectarse sin partir el backbone.
    ("mejor_modelo_convnext_tiny_min5pct9.pth", ("convnext_tiny", "min5pct9")),
    # Formato antiguo, sin configuración de clases en el nombre.
    ("mejor_modelo_densenet121.pth", ("densenet121", None)),
    # Backbone con guion bajo sin sufijo de config conocido → class_config None.
    ("mejor_modelo_efficientnet_b0.pth", ("efficientnet_b0", None)),
])
def test_parse_checkpoint_filename_casos(nombre, esperado):
    from src.models import parse_checkpoint_filename
    assert parse_checkpoint_filename(nombre) == esperado


def test_parse_checkpoint_filename_ignora_directorio_y_prefijos():
    from src.models import parse_checkpoint_filename
    assert parse_checkpoint_filename("models/_candidato_vgg16_full13.pth") == ("vgg16", "full13")
    assert parse_checkpoint_filename(
        "models\\mejor_modelo_densenet121_full13_subset.pth"
    ) == ("densenet121", "full13")
