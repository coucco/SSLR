import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


def tc_penalty(z_parts):
    """Оригинальная функция из Colab"""
    z_cat = torch.cat(list(z_parts.values()), dim=1)
    z_norm = (z_cat - z_cat.mean(0)) / (z_cat.std(0) + 1e-8)
    B = z_cat.size(0)
    corr = (z_norm.T @ z_norm) / B
    D = corr.size(0)
    mask = ~torch.eye(D, dtype=torch.bool, device=corr.device)
    return corr[mask].abs().mean()


def kl_div(enc):
    """Оригинальная функция из Colab"""
    kl = 0.0
    for k in enc['mu']:
        mu, lv = enc['mu'][k], enc['lv'][k]
        kl += -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(dim=1).mean()
    return kl


def total_loss(out, z_mv_target, x_flat, energy, cfg, beta):
    """Оригинальная функция total_loss из Colab"""
    B = x_flat.size(0)
    T = energy.size(1)

    recon_loss = F.mse_loss(out['z_mv_hat'], z_mv_target)
    kl = kl_div(out['enc'])
    energy_loss = F.mse_loss(out['e_pred'], energy)

    x3d = x_flat.view(B, T, 128, 3)
    xr = x3d[:, :, :, 0].reshape(B, -1)
    xh = x3d[:, :, :, 1].reshape(B, -1)
    xm = x3d[:, :, :, 2].reshape(B, -1)
    role_loss = (
        F.binary_cross_entropy_with_logits(out['r_pred'], xr, reduction='mean') +
        F.binary_cross_entropy_with_logits(out['h_pred'], xh, reduction='mean') +
        F.binary_cross_entropy_with_logits(out['m_pred'], xm, reduction='mean')
    ) / 3.0

    dis_loss = tc_penalty(out['z_parts'])

    total = (
        cfg['lambda_recon'] * recon_loss +
        beta * kl +
        cfg['lambda_energy'] * energy_loss +
        cfg['lambda_role'] * role_loss +
        cfg['lambda_dis'] * dis_loss
    )

    parts = dict(
        recon=recon_loss.item(), kl=kl.item(),
        energy=energy_loss.item(), role=role_loss.item(),
        dis=dis_loss.item(), total=total.item()
    )
    return total, parts


def beta_schedule(epoch, cfg):
    """Оригинальная функция из Colab"""
    w = cfg['kl_warmup']
    return min(cfg['lambda_kl'], cfg['lambda_kl'] * epoch / w)


def run_epoch(loader, model, optimizer, cfg, beta, device, train=True):
    """Оригинальная функция run_epoch из Colab"""
    if train:
        model.train()
    else:
        model.eval()
    
    sums = {k: 0.0 for k in ['total', 'recon', 'kl', 'energy', 'role', 'dis']}
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for z_mv, x_flat, energy in loader:
            z_mv = z_mv.to(device)
            x_flat = x_flat.to(device)
            energy = energy.to(device)

            out = model(x_flat)
            loss, parts = total_loss(out, z_mv, x_flat, energy, cfg, beta)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            for k, v in parts.items():
                sums[k] += v

    n = len(loader)
    return {k: v / n for k, v in sums.items()}


def train(model, train_loader, val_loader, cfg, device):
    """Оригинальный цикл обучения из Colab"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['n_epochs'], eta_min=1e-5)

    history = {k: [] for k in ['train_total', 'train_recon', 'train_kl', 'train_energy', 'train_role', 'train_dis',
                                'val_total', 'val_recon', 'val_kl', 'val_energy', 'val_role', 'val_dis']}
    best_val = float('inf')

    print('Starting training...')
    for epoch in range(1, cfg['n_epochs'] + 1):
        beta = beta_schedule(epoch, cfg)
        t0 = time.time()

        tr = run_epoch(train_loader, model, optimizer, cfg, beta, device, train=True)
        vl = run_epoch(val_loader, model, optimizer, cfg, beta, device, train=False)
        scheduler.step()

        for k, v in tr.items():
            history[f'train_{k}'].append(v)
        for k, v in vl.items():
            history[f'val_{k}'].append(v)

        if vl['total'] < best_val:
            best_val = vl['total']
            torch.save(dict(
                epoch=epoch, cfg=cfg,
                model_state=model.state_dict(),
                best_val=best_val,
            ), cfg['checkpoint_dir'] / 'best.pt')

        if epoch % 5 == 0 or epoch == 1:
            print(f'Ep {epoch:3d}/{cfg["n_epochs"]} β={beta:.3f} | '
                  f'tr: {tr["total"]:.4f} (rec={tr["recon"]:.4f} '
                  f'kl={tr["kl"]:.3f} role={tr["role"]:.4f}) | '
                  f'val: {vl["total"]:.4f} | {time.time() - t0:.1f}s')

    with open(cfg['checkpoint_dir'] / 'history.json', 'w') as f:
        json.dump(history, f)
    print(f'\nBest val loss: {best_val:.4f}')

    return history
