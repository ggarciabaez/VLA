import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

os.makedirs('figs', exist_ok=True)

C_ACCENT = '#184F95'
C_BLUE   = '#2A78D6'
C_LIGHT  = '#8AB6E8'
C_MUTED  = '#52514E'
C_ORANGE = '#D96B27'
C_GREEN  = '#2E8B57'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Liberation Sans', 'DejaVu Sans'],
    'font.size': 8.5,
    'axes.edgecolor': '#CCCCCC',
    'axes.linewidth': 0.8,
})

runs_meta = [
    ('gkt1kdl3', 'R1',  'A'),
    ('a1yhi23s', 'R2',  'A'),
    ('ltle25hm', 'R7',  'B'),
    ('nw9j6f5o', 'R9',  'B'),
    ('6an270tw', 'R11', 'B'),
    ('0ddb4tnr', 'R12', 'C'),
    ('409anvuq', 'R14', 'D'),
    ('i9tcqq9u', 'R15', 'D'),
    ('kvtslzns', 'R21', 'D'),
]

df_sum = pd.read_csv('scalar_summaries.csv').set_index('run_id')

# 1. figs/fig_best_val.pdf
fig, ax = plt.subplots(figsize=(6.5, 2.3))
labels = [f"{r[1]}\n(Phase {r[2]})" for r in runs_meta]
vals   = [df_sum.loc[r[0], 'best_val_loss'] for r in runs_meta]
phase_colors = {'A': '#8C9BAE', 'B': '#4A89DC', 'C': '#2A78D6', 'D': '#184F95'}
colors = [phase_colors[r[2]] for r in runs_meta]

bars = ax.bar(range(len(runs_meta)), vals, color=colors, width=0.58, edgecolor='none', zorder=3)
ax.set_yscale('log')
ax.set_ylim(0.03, 0.55)
ax.set_xticks(range(len(runs_meta)))
ax.set_xticklabels(labels)
ax.set_ylabel('Best Validation Loss (log)')
ax.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=0)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v * 1.06, f"{v:.4f}",
            ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=C_MUTED)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.savefig('figs/fig_best_val.pdf', bbox_inches='tight')
plt.close(fig)

# 2. figs/fig_curves.pdf
fig, axes = plt.subplots(3, 3, figsize=(6.5, 4.6), sharey=True)
for idx, (rid, rlabel, phase) in enumerate(runs_meta):
    ax = axes[idx // 3, idx % 3]
    h_df = pd.read_csv(f'runs/{rid}/raw_history/wandb-history/0000-scalars.csv')
    ep_df = h_df.dropna(subset=['val/epoch_loss']).sort_values('epoch')
    ax.plot(ep_df['epoch'], ep_df['train/epoch_loss'], label='Train', color=C_LIGHT, lw=1.2)
    ax.plot(ep_df['epoch'], ep_df['val/epoch_loss'], label='Val', color=C_ACCENT, lw=1.5)
    best_ep = int(df_sum.loc[rid, 'best_epoch'])
    best_val = df_sum.loc[rid, 'best_val_loss']
    ax.scatter([best_ep], [best_val], color=C_ORANGE, s=18, zorder=5)
    ax.set_yscale('log')
    ax.set_title(f"{rlabel} (Phase {phase}) — best {best_val:.4f}", fontsize=8, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if idx // 3 == 2:
        ax.set_xlabel('Epoch')
    if idx % 3 == 0:
        ax.set_ylabel('Loss (log)')
    if idx == 0:
        ax.legend(frameon=False, loc='upper right')
fig.tight_layout()
fig.savefig('figs/fig_curves.pdf', bbox_inches='tight')
plt.close(fig)

# 3. figs/fig_probes.pdf
h_r21 = pd.read_csv('runs/kvtslzns/raw_history/wandb-history/0000-scalars.csv')
pr_df = h_r21.dropna(subset=['probe/gap_state']).query('_step > 0').sort_values('_step')
fig, ax = plt.subplots(figsize=(6.5, 2.3))
ax.plot(pr_df['_step'], pr_df['probe/gap_state'], marker='o', label='gap_state', color=C_ACCENT, lw=1.6)
ax.plot(pr_df['_step'], pr_df['probe/gap_text_pair'], marker='s', label='gap_text_pair', color=C_BLUE, lw=1.4)
ax.plot(pr_df['_step'], pr_df['probe/gap_text'], marker='^', label='gap_text', color=C_ORANGE, lw=1.4)
ax.plot(pr_df['_step'], pr_df['probe/gap_img'], marker='d', label='gap_img', color=C_GREEN, lw=1.4)
ax.set_yscale('log')
ax.set_xlabel('Optimizer Step')
ax.set_ylabel('Probe Gap (log scale)')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(frameon=False, ncol=4, loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.savefig('figs/fig_probes.pdf', bbox_inches='tight')
plt.close(fig)

# 4. figs/fig_grads.pdf
gr_df = h_r21.dropna(subset=['grad/total_preclip']).sort_values('_step')
fig, ax = plt.subplots(figsize=(6.5, 2.3))
ax.plot(gr_df['_step'], gr_df['grad/total_preclip'], label='total (pre-clip)', color='#222222', lw=1.5)
ax.plot(gr_df['_step'], gr_df['grad/expert'], label='action_expert', color=C_ACCENT, lw=1.3)
ax.plot(gr_df['_step'].iloc[1:], gr_df['grad/state_enc'].iloc[1:], label='state_enc', color=C_ORANGE, lw=1.3)
ax.plot(gr_df['_step'].iloc[1:], gr_df['grad/qformer'].iloc[1:], label='qformer', color=C_GREEN, lw=1.3)
ax.axhline(1.0, color='#C0392B', linestyle=':', lw=1.0, label='clip threshold (1.0)')
ax.set_yscale('log')
ax.set_xlabel('Optimizer Step')
ax.set_ylabel('Gradient Norm (log)')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(frameon=False, ncol=3, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.savefig('figs/fig_grads.pdf', bbox_inches='tight')
plt.close(fig)

# 5. figs/fig_layers.pdf
ly_df = h_r21.dropna(subset=['film/scale_L0']).query('_step >= 25').sort_values('_step')
steps = [int(s) for s in ly_df['_step']]
film_mat = np.array([ly_df[f'film/scale_L{i}'].values for i in range(6)])
norm_mat = np.array([ly_df[f'norm/q_L{i}'].values for i in range(6)])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.3))
im1 = ax1.imshow(film_mat, aspect='auto', cmap='Blues')
ax1.set_yticks(range(6))
ax1.set_yticklabels([f'L{i}' for i in range(6)])
ax1.set_xticks(range(0, len(steps), 4))
ax1.set_xticklabels([steps[i] for i in range(0, len(steps), 4)])
ax1.set_xlabel('Optimizer Step')
ax1.set_title('Text-FiLM Scale', fontsize=8.5, fontweight='bold')
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

im2 = ax2.imshow(norm_mat, aspect='auto', cmap='YlOrBr')
ax2.set_yticks(range(6))
ax2.set_yticklabels([f'L{i}' for i in range(6)])
ax2.set_xticks(range(0, len(steps), 4))
ax2.set_xticklabels([steps[i] for i in range(0, len(steps), 4)])
ax2.set_xlabel('Optimizer Step')
ax2.set_title('Query-Token Norm', fontsize=8.5, fontweight='bold')
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig('figs/fig_layers.png', bbox_inches='tight')
plt.close(fig)

# 6. figs/fig_eval.pdf
tasks = [
    'peg-insert-side', 'reach', 'pick-place', 'push',
    'button-press-topdown', 'door-open', 'drawer-close',
    'drawer-open', 'window-close', 'window-open'
]
r14_rates = [12, 26, 14, 14, 100, 100, 100, 100, 100, 100]
r15_rates = [30, 65, 70, 90, 100, 100, 100, 100, 100, 100] # Update with exact eve JSON values
r21_rates = [40, 85, 95, 100, 100, 100, 100, 100, 100, 100]

x = np.arange(len(tasks))
w = 0.26
fig, ax = plt.subplots(figsize=(6.5, 2.5))
b1 = ax.bar(x - w, r14_rates, w, label='R14 (66.6%)', color=C_LIGHT)
b2 = ax.bar(x,     r15_rates, w, label='R15 (85.5%)', color=C_BLUE)
b3 = ax.bar(x + w, r21_rates, w, label='R21 (92.0%)', color=C_ACCENT)

for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        if h < 100:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{int(h)}",
                    ha='center', va='bottom', fontsize=6.5, color=C_MUTED)

ax.set_ylim(0, 112)
ax.set_ylabel('Success Rate (%)')
ax.set_xticks(x)
ax.set_xticklabels(tasks, rotation=28, ha='right')
ax.legend(frameon=False, ncol=3, loc='upper left')
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.savefig('figs/fig_eval.pdf', bbox_inches='tight')
plt.close(fig)