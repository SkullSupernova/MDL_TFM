import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.train import train_model


# =========================================================
# Helpers
# =========================================================

def _make_loader(n=16, in_features=4, num_classes=3, batch_size=4):
    X = torch.randn(n, in_features)
    Y = torch.randint(0, 2, (n, num_classes)).float()
    return DataLoader(TensorDataset(X, Y), batch_size=batch_size)


def _make_model(in_features=4, num_classes=3):
    return nn.Sequential(
        nn.Linear(in_features, 8),
        nn.ReLU(),
        nn.Linear(8, num_classes),
    )


# =========================================================
# train_model — happy path
# =========================================================

def test_bucle_entrenamiento_dos_epocas_completa(tmp_path):
    model = _make_model()
    loader = _make_loader()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    history, _ = train_model(
        model, loader, loader, criterion, optimizer, scheduler,
        num_epochs=2, device=torch.device("cpu"),
        save_path=str(tmp_path / "ckpt.pth"),
    )
    assert len(history['train_loss']) == 2
    assert len(history['val_loss']) == 2


def test_history_contiene_todas_las_claves(tmp_path):
    model = _make_model()
    loader = _make_loader()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    history, _ = train_model(
        model, loader, loader, criterion, optimizer, scheduler,
        num_epochs=1, device=torch.device("cpu"),
        save_path=str(tmp_path / "ckpt.pth"),
    )
    assert all(k in history for k in ('train_loss', 'val_loss', 'val_acc', 'val_f1'))


def test_modelo_devuelto_es_instancia_nn_module(tmp_path):
    model = _make_model()
    loader = _make_loader()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    _, best_model = train_model(
        model, loader, loader, criterion, optimizer, scheduler,
        num_epochs=1, device=torch.device("cpu"),
        save_path=str(tmp_path / "ckpt.pth"),
    )
    assert isinstance(best_model, nn.Module)


# =========================================================
# train_model — edge cases
# =========================================================

def test_numero_epocas_coincide_exactamente_con_solicitadas(tmp_path):
    # Con mejora continua (lr suficiente), el bucle completa todas las épocas indicadas
    model = _make_model()
    loader = _make_loader(n=32)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    history, _ = train_model(
        model, loader, loader, criterion, optimizer, scheduler,
        num_epochs=3, device=torch.device("cpu"),
        save_path=str(tmp_path / "ckpt.pth"),
    )
    assert len(history['train_loss']) == 3
    assert len(history['val_loss']) == 3


def test_train_loss_decrece_tras_varias_epocas(tmp_path):
    # Con lr suficientemente alto, la pérdida debe reducirse en varias épocas
    model = _make_model()
    loader = _make_loader(n=32)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    history, _ = train_model(
        model, loader, loader, criterion, optimizer, scheduler,
        num_epochs=5, device=torch.device("cpu"),
        save_path=str(tmp_path / "ckpt.pth"),
    )
    assert history['train_loss'][-1] < history['train_loss'][0]


# =========================================================
# train_model — gestión de errores
# =========================================================

def test_checkpoint_guardado_en_ruta_indicada(tmp_path):
    model = _make_model()
    loader = _make_loader()
    save_path = str(tmp_path / "subdir" / "modelo.pth")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    import os
    os.makedirs(str(tmp_path / "subdir"))

    train_model(
        model, loader, loader, criterion, optimizer, scheduler,
        num_epochs=1, device=torch.device("cpu"), save_path=save_path,
    )
    assert os.path.exists(save_path)


# =========================================================
# train_model — reanudación (checkpoints reanudables)
# =========================================================

def _componentes():
    model = _make_model()
    loader = _make_loader()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    return model, loader, criterion, optimizer, scheduler


def test_resume_continua_desde_checkpoint(tmp_path):
    import os
    model, loader, criterion, optimizer, scheduler = _componentes()
    resume_path = str(tmp_path / "ckpt_resume.pth")
    # Checkpoint que simula que la época 0 ya se completó.
    torch.save({
        "epoch": 0,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "history": {"train_loss": [1.0], "val_loss": [1.0], "val_acc": [0.0],
                    "val_f1": [0.0], "val_auroc": [0.5]},
        "best_metric": 0.5,
        "best_model_state": model.state_dict(),
        "es_best_loss": 1.0, "es_counter": 0, "es_early_stop": False,
    }, resume_path)

    history, _ = train_model(
        model, loader, loader, criterion, optimizer, scheduler,
        num_epochs=3, device=torch.device("cpu"),
        save_path=str(tmp_path / "best.pth"),
        resume_path=resume_path, resume=True,
    )
    # 1 época del checkpoint + 2 nuevas (épocas 1 y 2) = 3 entradas.
    assert len(history["train_loss"]) == 3
    # Al terminar correctamente, el checkpoint reanudable se elimina.
    assert not os.path.exists(resume_path)


def test_resume_sin_checkpoint_empieza_de_cero(tmp_path):
    model, loader, criterion, optimizer, scheduler = _componentes()
    history, _ = train_model(
        model, loader, loader, criterion, optimizer, scheduler,
        num_epochs=2, device=torch.device("cpu"),
        save_path=str(tmp_path / "best.pth"),
        resume_path=str(tmp_path / "no_existe.pth"), resume=True,
    )
    assert len(history["train_loss"]) == 2
