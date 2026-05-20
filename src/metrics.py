import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path


def tc_penalty(z_parts):
    """Оригинальная функция из Colab"""
    z_cat = torch.cat(list(z_parts.values()), dim=1)
    z_norm = (z_cat - z_cat.mean(0)) / (z_cat.std(0) + 1e-8)
    B = z_cat.size(0)
    corr = (z_norm.T @ z_norm) / B
    D = corr.size(0)
    mask = ~torch.eye(D, dtype=torch.bool, device=corr.device)
    return corr[mask].abs().mean()


def compute_all_metrics(controller, test_loader, cfg, device):
    """Оригинальная логика вычисления метрик из Colab"""
    controller.eval()
    metrics = dict(
        z_recon_mse=[], energy_corr=[],
        inter_track_corr=[], tc_score=[],
    )
    kl_per_comp = {k: [] for k in ['struct', 'energy', 'rhythm', 'harmony', 'melody']}

    with torch.no_grad():
        for z_mv, x_flat, energy in test_loader:
            z_mv = z_mv.to(device)
            x_flat = x_flat.to(device)
            energy = energy.to(device)

            out = controller(x_flat)

            mse = F.mse_loss(out['z_mv_hat'], z_mv).item()
            metrics['z_recon_mse'].append(mse)

            e_pred = out['e_pred'].cpu().numpy()
            e_true = energy.cpu().numpy()
            for b in range(e_pred.shape[0]):
                if e_true[b].std() > 1e-6 and e_pred[b].std() > 1e-6:
                    metrics['energy_corr'].append(float(np.corrcoef(e_true[b], e_pred[b])[0, 1]))

            B = x_flat.size(0)
            T_FRAMES = cfg['t_frames']
            h_out = torch.sigmoid(out['h_pred']).cpu().numpy().reshape(B, T_FRAMES, 128)
            m_out = torch.sigmoid(out['m_pred']).cpu().numpy().reshape(B, T_FRAMES, 128)
            for b in range(B):
                hd = h_out[b].sum(axis=1)
                md = m_out[b].sum(axis=1)
                if hd.std() > 1e-6 and md.std() > 1e-6:
                    metrics['inter_track_corr'].append(float(np.corrcoef(hd, md)[0, 1]))

            tc = tc_penalty(out['z_parts']).item()
            metrics['tc_score'].append(tc)

            for comp in kl_per_comp:
                mu = out['enc']['mu'][comp]
                lv = out['enc']['lv'][comp]
                kl_c = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean().item()
                kl_per_comp[comp].append(kl_c)

    final = {k: float(np.mean(v)) for k, v in metrics.items()}
    final['kl_per_component'] = {k: float(np.mean(v)) for k, v in kl_per_comp.items()}
    return final


def plot_training_curves(history_path, output_dir):
    """Оригинальная визуализация из Colab"""
    import json
    import matplotlib.pyplot as plt

    with open(history_path, 'r') as f:
        history = json.load(f)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    keys = ['total', 'recon', 'kl', 'energy', 'role', 'dis']
    titles = ['Total Loss', 'z Recon (MSE)', 'KL Divergence',
              'Energy Loss', 'Role Aux Loss', 'TC Penalty']

    for ax, key, title in zip(axes.flat, keys, titles):
        ax.plot(history.get(f'train_{key}', []), label='train', color='#3498db', lw=1.5)
        ax.plot(history.get(f'val_{key}', []), label='val', color='#e74c3c', lw=1.5)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xlabel('Epoch')
        ax.grid(alpha=0.3)

    plt.suptitle('Structured Controller — Training Curves', fontsize=14)
    plt.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()


def plot_kl_per_component(kl_dict, output_dir):
    """Оригинальная визуализация из Colab"""
    import matplotlib.pyplot as plt

    colors = ['#9b59b6', '#f39c12', '#e74c3c', '#2ecc71', '#3498db']

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(kl_dict.keys(), kl_dict.values(), color=colors, edgecolor='white', width=0.5)
    ax.axhline(0.1, color='gray', linestyle='--', lw=1, label='collapse threshold')
    ax.set_title('KL Divergence per Latent Component\n(values > 0.1 indicate active, non-collapsed components)', fontsize=11)
    ax.set_ylabel('Mean KL')
    ax.legend()
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'kl_per_component.png', dpi=150)
    plt.close()


def plot_pca_latent_space(controller, test_loader, device, output_dir):
    """Оригинальная визуализация PCA из Colab"""
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    z_all_parts = {k: [] for k in ['struct', 'energy', 'rhythm', 'harmony', 'melody']}
    controller.eval()
    with torch.no_grad():
        for _, x_flat, _ in test_loader:
            enc = controller.encode(x_flat.to(device))
            for k in z_all_parts:
                z_all_parts[k].append(enc['mu'][k].cpu().numpy())
            if sum(len(v) for v in z_all_parts.values()) > 5 * 1000:
                break

    z_concat = np.concatenate([np.concatenate(z_all_parts[k], 0) for k in z_all_parts], axis=1)[:1000]
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_concat)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=np.arange(len(z_2d)), cmap='viridis', alpha=0.4, s=8)
    plt.colorbar(sc, label='Sample index')
    ax.set_title('Structured Latent Space (PCA)')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'latent_pca.png', dpi=150)
    plt.close()
