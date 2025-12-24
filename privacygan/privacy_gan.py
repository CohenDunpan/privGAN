# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import warnings
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

from privacygan.mnist.mnist_gan import (
    MNIST_Discriminator,
    MNIST_DiscriminatorPrivate,
    MNIST_Generator,
    make_optimizer,
)

warnings.filterwarnings("ignore")


def _get_device(device: Optional[torch.device] = None) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(data: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.float32, device=device)


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


##########################################  GANs #############################################


def SimpGAN(
    X_train: np.ndarray,
    generator: nn.Module = None,
    discriminator: nn.Module = None,
    randomDim: int = 100,
    epochs: int = 200,
    batchSize: int = 128,
    lr: float = 0.0002,
    beta1: float = 0.5,
    verbose: int = 1,
    lSmooth: float = 0.9,
    SplitTF: bool = False,
    device: Optional[torch.device] = None,
):
    """Single GAN training loop implemented in PyTorch."""

    device = _get_device(device)
    generator = generator or MNIST_Generator(randomDim=randomDim)
    discriminator = discriminator or MNIST_Discriminator()
    generator.to(device)
    discriminator.to(device)

    g_opt = make_optimizer(generator, lr=lr, beta1=beta1)
    d_opt = make_optimizer(discriminator, lr=lr, beta1=beta1)
    bce = nn.BCELoss()

    dataset = TensorDataset(_to_tensor(X_train, device))
    loader = DataLoader(dataset, batch_size=batchSize, shuffle=True, drop_last=True)

    dLosses: List[float] = []
    gLosses: List[float] = []

    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', len(loader))

    for e in range(1, epochs + 1):
        g_t: List[float] = []
        d_t: List[float] = []

        for i, (real_batch,) in enumerate(loader):
            noise = torch.randn(batchSize, randomDim, device=device)
            fake_batch = generator(noise)

            # discriminator step
            _set_requires_grad(discriminator, True)
            d_opt.zero_grad()
            real_labels = torch.full((batchSize, 1), lSmooth, device=device)
            fake_labels = torch.zeros(batchSize, 1, device=device)

            d_loss_real = bce(discriminator(real_batch), real_labels)
            d_loss_fake = bce(discriminator(fake_batch.detach()), fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_opt.step()
            _set_requires_grad(discriminator, False)

            # generator step
            g_opt.zero_grad()
            gen_labels = torch.ones(batchSize, 1, device=device)
            g_loss = bce(discriminator(fake_batch), gen_labels)
            g_loss.backward()
            g_opt.step()

            if verbose == 1:
                print(
                    f"epoch = {e}/{epochs}, batch = {i+1}/{len(loader)}, d_loss={d_loss.item():.3f}, g_loss={g_loss.item():.3f}",
                    end='\r',
                )

            d_t.append(d_loss.item())
            g_t.append(g_loss.item())

        dLosses.append(float(np.mean(d_t)))
        gLosses.append(float(np.mean(g_t)))

        if verbose == 1:
            print(
                f"epoch = {e}/{epochs}, d_loss={dLosses[-1]:.3f}, g_loss={gLosses[-1]:.3f}"
            )

    return generator, discriminator, dLosses, gLosses


def TrainDiscriminator(
    X_train: np.ndarray,
    y_train: np.ndarray,
    discriminator: nn.Module = None,
    epochs: int = 200,
    batchSize: int = 128,
    lr: float = 0.0002,
    beta1: float = 0.5,
    device: Optional[torch.device] = None,
):
    """Train the privacy discriminator (multi-class classifier)."""

    device = _get_device(device)
    discriminator = discriminator or MNIST_DiscriminatorPrivate(OutSize=len(np.unique(y_train)))
    discriminator.to(device)

    opt = make_optimizer(discriminator, lr=lr, beta1=beta1)
    ce = nn.CrossEntropyLoss()

    dataset = TensorDataset(_to_tensor(X_train, device), torch.tensor(y_train, dtype=torch.long, device=device))
    loader = DataLoader(dataset, batch_size=batchSize, shuffle=True, drop_last=True)

    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            logits = discriminator(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()

    return discriminator


def privGAN(
    X_train: np.ndarray,
    generators: Optional[Sequence[nn.Module]] = None,
    discriminators: Optional[Sequence[nn.Module]] = None,
    pDisc: Optional[nn.Module] = None,
    randomDim: int = 100,
    disc_epochs: int = 50,
    epochs: int = 200,
    dp_delay: int = 100,
    batchSize: int = 128,
    lr: float = 0.0002,
    beta1: float = 0.5,
    verbose: int = 1,
    lSmooth: float = 0.95,
    privacy_ratio: float = 1.0,
    SplitTF: bool = False,
    device: Optional[torch.device] = None,
):
    """PrivGAN training loop in PyTorch."""

    device = _get_device(device)

    if generators is None:
        generators = [MNIST_Generator(randomDim=randomDim), MNIST_Generator(randomDim=randomDim)]
    if discriminators is None:
        discriminators = [MNIST_Discriminator(), MNIST_Discriminator()]
    if len(generators) != len(discriminators):
        raise ValueError('Different number of generators and discriminators')

    n_reps = len(generators)
    if n_reps == 1:
        raise ValueError('You cannot have only one generator-discriminator pair')

    pDisc = pDisc or MNIST_DiscriminatorPrivate(OutSize=n_reps)

    for g, d in zip(generators, discriminators):
        g.to(device)
        d.to(device)
    pDisc.to(device)

    g_opts = [make_optimizer(g, lr=lr, beta1=beta1) for g in generators]
    d_opts = [make_optimizer(d, lr=lr, beta1=beta1) for d in discriminators]
    p_opt = make_optimizer(pDisc, lr=lr, beta1=beta1)

    bce = nn.BCELoss()
    ce = nn.CrossEntropyLoss()

    # split dataset across generators
    X_splits: List[torch.Tensor] = []
    y_train = []
    t = len(X_train) // n_reps
    for i in range(n_reps):
        if i < n_reps - 1:
            X_splits.append(_to_tensor(X_train[i * t : (i + 1) * t], device))
            y_train.extend([i] * t)
        else:
            X_splits.append(_to_tensor(X_train[i * t :], device))
            y_train.extend([i] * len(X_train[i * t :]))
    y_train = np.array(y_train)

    # pretrain privacy discriminator
    TrainDiscriminator(X_train, y_train, discriminator=pDisc, epochs=disc_epochs, batchSize=batchSize, lr=lr, beta1=beta1, device=device)
    with torch.no_grad():
        logits = pDisc(_to_tensor(X_train, device))
        yp = logits.argmax(dim=1).cpu().numpy()
        print('dp-Accuracy:', np.mean(y_train == yp))

    # Ensure at least one batch per epoch to avoid empty loops when t < batchSize
    batchCount = max(1, int(np.ceil(t / batchSize)))
    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)

    dLosses = np.zeros((n_reps, epochs))
    dpLosses = np.zeros(epochs)
    gLosses = np.zeros(epochs)

    for e in range(epochs):
        d_t = np.zeros((n_reps, batchCount))
        dp_t = np.zeros(batchCount)
        g_t = np.zeros(batchCount)

        for i in range(batchCount):
            noise = torch.randn(batchSize, randomDim, device=device)
            generatedImages: List[torch.Tensor] = []
            yDis2: List[int] = []
            yDis2f: List[np.ndarray] = []

            # discriminator updates
            for j in range(n_reps):
                real_batch = X_splits[j][torch.randint(0, len(X_splits[j]), (batchSize,), device=device)]
                fake_batch = generators[j](noise)
                generatedImages.append(fake_batch)

                real_labels = torch.full((batchSize, 1), lSmooth, device=device)
                fake_labels = torch.zeros(batchSize, 1, device=device)

                _set_requires_grad(discriminators[j], True)
                d_opts[j].zero_grad()
                d_loss_real = bce(discriminators[j](real_batch), real_labels)
                d_loss_fake = bce(discriminators[j](fake_batch.detach()), fake_labels)
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_opts[j].step()
                _set_requires_grad(discriminators[j], False)

                d_t[j, i] = d_loss.item()
                labels_other = list(range(n_reps))
                labels_other.remove(j)
                yDis2.extend([j] * batchSize)
                yDis2f.append(np.random.choice(labels_other, size=batchSize))

            yDis2_arr = np.array(yDis2)
            all_generated = torch.cat(generatedImages, dim=0)

            # privacy discriminator update
            if e >= dp_delay:
                _set_requires_grad(pDisc, True)
                p_opt.zero_grad()
                logits = pDisc(all_generated.detach())
                dp_loss = ce(logits, torch.tensor(yDis2_arr, device=device))
                dp_loss.backward()
                p_opt.step()
                _set_requires_grad(pDisc, False)
                dp_t[i] = dp_loss.item()

            # generator updates
            _set_requires_grad(pDisc, False)
            for j in range(n_reps):
                g_opts[j].zero_grad()
                adv_labels = torch.ones(batchSize, 1, device=device)
                adv_loss = bce(discriminators[j](generatedImages[j]), adv_labels)
                priv_targets = torch.tensor(yDis2f[j], device=device)
                priv_loss = ce(pDisc(generatedImages[j]), priv_targets)
                g_loss = adv_loss + privacy_ratio * priv_loss
                g_loss.backward()
                g_opts[j].step()
                g_t[i] += g_loss.item()

            if verbose == 1:
                print(f"epoch = {e}/{epochs}, batch = {i+1}/{batchCount}", end='\r')

        dLosses[:, e] = np.mean(d_t, axis=1)
        dpLosses[e] = np.mean(dp_t) if np.any(dp_t) else 0.0
        gLosses[e] = np.mean(g_t)

        if verbose == 1:
            print('epoch =', e)
            print('dLosses =', np.mean(d_t, axis=1))
            print('dpLosses =', dpLosses[e])
            print('gLosses =', gLosses[e])
            with torch.no_grad():
                yp = pDisc(all_generated).argmax(dim=1).cpu().numpy()
                print('dp-Accuracy:', np.mean(yDis2_arr == yp))

    return generators, discriminators, pDisc, dLosses, dpLosses, gLosses


######################################### Ancillary functions ################################


def DisplayImages(
    generator: nn.Module,
    randomDim: int = 100,
    NoImages: int = 100,
    figSize: Tuple[int, int] = (10, 10),
    TargetShape: Tuple[int, ...] = (28, 28),
    device: Optional[torch.device] = None,
):
    # check figure size
    if (len(figSize) != 2) or (figSize[0] * figSize[1] < NoImages):
        print('Invalid Figure Size')
        return

    device = _get_device(device)
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(NoImages, randomDim, device=device)
        generatedImages = generator(noise).cpu().numpy()

    TargetShape = tuple([NoImages] + list(TargetShape))
    generatedImages = generatedImages.reshape(TargetShape)

    for i in range(generatedImages.shape[0]):
        plt.subplot(figSize[0], figSize[1], i + 1)
        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()


############################################ Attacks #########################################


def _predict_disc(discriminator: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    discriminator.eval()
    with torch.no_grad():
        preds = discriminator(_to_tensor(X, device)).cpu().numpy().squeeze()
    return preds


def WBattack(X: np.ndarray, X_comp: np.ndarray, discriminator: nn.Module, device: Optional[torch.device] = None):
    device = _get_device(device)
    Dat = np.concatenate([X, X_comp])
    p = _predict_disc(discriminator, Dat, device)
    In = np.argsort(-p)[: len(X)]
    Accuracy = np.sum(1.0 * (In < len(X))) / len(X)
    print('White-box attack accuracy:', Accuracy)
    return Accuracy


def WBattack_priv(
    X: np.ndarray,
    X_comp: np.ndarray,
    discriminators: Sequence[nn.Module],
    device: Optional[torch.device] = None,
):
    device = _get_device(device)
    Dat = np.concatenate([X, X_comp])
    Pred = [
        _predict_disc(discriminators[i], Dat, device)
        for i in range(len(discriminators))
    ]
    p_mean = np.mean(Pred, axis=0)
    p_max = np.max(Pred, axis=0)

    In_mean = np.argsort(-p_mean)[: len(X)]
    In_max = np.argsort(-p_max)[: len(X)]

    Acc_max = np.sum(1.0 * (In_max < len(X))) / len(X)
    Acc_mean = np.sum(1.0 * (In_mean < len(X))) / len(X)

    print('White-box attack accuracy (max):', Acc_max)
    print('White-box attack accuracy (mean):', Acc_mean)
    return Acc_max, Acc_mean


def WBattack_TVD(X: np.ndarray, X_comp: np.ndarray, discriminator: nn.Module, device: Optional[torch.device] = None):
    device = _get_device(device)
    n1, _ = np.histogram(_predict_disc(discriminator, X, device), bins=50, density=True, range=[0, 1])
    n2, _ = np.histogram(_predict_disc(discriminator, X_comp, device), bins=50, density=True, range=[0, 1])
    tvd = 0.5 * np.linalg.norm(n1 - n2, 1) / 50.0
    print('Total Variational Distance:', tvd)
    return tvd


def WBattack_TVD_priv(
    X: np.ndarray,
    X_comp: np.ndarray,
    discriminators: Sequence[nn.Module],
    device: Optional[torch.device] = None,
):
    device = _get_device(device)
    tvd = []
    for disc in discriminators:
        n1, _ = np.histogram(_predict_disc(disc, X, device), bins=50, density=True, range=[0, 1])
        n2, _ = np.histogram(_predict_disc(disc, X_comp, device), bins=50, density=True, range=[0, 1])
        tvd.append(0.5 * np.linalg.norm(n1 - n2, 1) / 50.0)
    print('Total Variational Distance - max:', max(tvd))
    print('Total Variational Distance - mean:', np.mean(tvd))
    return float(np.max(tvd)), float(np.mean(tvd))


def _generate_fake(generator: nn.Module, N: int, randomDim: int, device: torch.device) -> np.ndarray:
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(N, randomDim, device=device)
        X_fake = generator(noise).cpu().numpy()
    return X_fake


def MC_eps_attack(
    X: np.ndarray,
    X_comp: np.ndarray,
    X_ho: np.ndarray,
    generator: nn.Module,
    N: int = 100000,
    M: int = 100,
    n_pc: int = 40,
    reps: int = 10,
    randomDim: int = 100,
    device: Optional[torch.device] = None,
):
    device = _get_device(device)
    sh = int(np.prod(X.shape[1:]))
    X = np.reshape(X, (len(X), sh))
    X_comp = np.reshape(X_comp, (len(X_comp), sh))
    X_ho = np.reshape(X_ho, (len(X_ho), sh))

    pca = PCA(n_components=n_pc)
    pca.fit(X_ho)

    res = []
    for _ in range(reps):
        X_fake = _generate_fake(generator, N, randomDim, device)
        X_fake = np.reshape(X_fake, (len(X_fake), sh))
        X_fake_dr = pca.transform(X_fake)

        idx1 = np.random.randint(len(X), size=M)
        M_x = pca.transform(np.reshape(X[idx1, :], (len(idx1), sh)))
        M_xc = pca.transform(np.reshape(X_comp[idx1, :], (len(idx1), sh)))

        min_x = []
        min_xc = []
        for i in range(M):
            temp_x = np.tile(M_x[i, :], (len(X_fake_dr), 1))
            temp_xc = np.tile(M_xc[i, :], (len(X_fake_dr), 1))
            D_x = np.sqrt(np.sum((temp_x - X_fake_dr) ** 2, axis=1))
            D_xc = np.sqrt(np.sum((temp_xc - X_fake_dr) ** 2, axis=1))
            min_x.append(np.min(D_x))
            min_xc.append(np.min(D_xc))

        eps = np.median(min_x + min_xc)
        s_x = []
        s_xc = []
        for i in range(M):
            temp_x = np.tile(M_x[i, :], (len(X_fake_dr), 1))
            temp_xc = np.tile(M_xc[i, :], (len(X_fake_dr), 1))
            D_x = np.sqrt(np.sum((temp_x - X_fake_dr) ** 2, axis=1))
            D_xc = np.sqrt(np.sum((temp_xc - X_fake_dr) ** 2, axis=1))
            s_x.append(np.sum(D_x <= eps) / len(X_fake_dr))
            s_xc.append(np.sum(D_xc <= eps) / len(X_fake_dr))

        s_x_xc = np.array(s_x + s_xc)
        In = np.argsort(-s_x_xc)[:M]
        res.append(1 if np.sum(In < M) >= 0.5 * M else 0)

    return float(np.mean(res))


def MC_eps_attack_priv(
    X: np.ndarray,
    X_comp: np.ndarray,
    X_ho: np.ndarray,
    generators: Sequence[nn.Module],
    N: int = 100000,
    M: int = 100,
    n_pc: int = 40,
    reps: int = 10,
    randomDim: int = 100,
    device: Optional[torch.device] = None,
):
    device = _get_device(device)
    sh = int(np.prod(X.shape[1:]))
    X = np.reshape(X, (len(X), sh))
    X_comp = np.reshape(X_comp, (len(X_comp), sh))
    X_ho = np.reshape(X_ho, (len(X_ho), sh))

    pca = PCA(n_components=n_pc)
    pca.fit(X_ho)

    res = []
    for _ in range(reps):
        n_g = len(generators)
        X_fake_dr = []
        for j in range(n_g):
            X_fake = _generate_fake(generators[j], int(N / n_g), randomDim, device)
            X_fake = np.reshape(X_fake, (len(X_fake), sh))
            X_fake_dr.append(pca.transform(X_fake))
        X_fake_dr = np.vstack(X_fake_dr)

        idx1 = np.random.randint(len(X), size=M)
        M_x = pca.transform(np.reshape(X[idx1, :], (len(idx1), sh)))
        M_xc = pca.transform(np.reshape(X_comp[idx1, :], (len(idx1), sh)))

        min_x = []
        min_xc = []
        for i in range(M):
            temp_x = np.tile(M_x[i, :], (len(X_fake_dr), 1))
            temp_xc = np.tile(M_xc[i, :], (len(X_fake_dr), 1))
            D_x = np.sqrt(np.sum((temp_x - X_fake_dr) ** 2, axis=1))
            D_xc = np.sqrt(np.sum((temp_xc - X_fake_dr) ** 2, axis=1))
            min_x.append(np.min(D_x))
            min_xc.append(np.min(D_xc))

        eps = np.median(min_x + min_xc)
        s_x = []
        s_xc = []
        for i in range(M):
            temp_x = np.tile(M_x[i, :], (len(X_fake_dr), 1))
            temp_xc = np.tile(M_xc[i, :], (len(X_fake_dr), 1))
            D_x = np.sqrt(np.sum((temp_x - X_fake_dr) ** 2, axis=1))
            D_xc = np.sqrt(np.sum((temp_xc - X_fake_dr) ** 2, axis=1))
            s_x.append(np.sum(D_x <= eps) / len(X_fake_dr))
            s_xc.append(np.sum(D_xc <= eps) / len(X_fake_dr))

        s_x_xc = np.array(s_x + s_xc)
        In = np.argsort(-s_x_xc)[:M]
        res.append(1 if np.sum(In < M) >= 0.5 * M else 0)

    return float(np.mean(res))


__all__ = [
    'SimpGAN',
    'TrainDiscriminator',
    'privGAN',
    'DisplayImages',
    'WBattack',
    'WBattack_priv',
    'WBattack_TVD',
    'WBattack_TVD_priv',
    'MC_eps_attack',
    'MC_eps_attack_priv',
]
