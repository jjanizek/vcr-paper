#!/usr/bin/env python
"""
Composite figure generator for VCR validation results.

Usage:
    python plot_vcr_validation_composite.py --results_dir ./multi_feature_results_v2_OpenFlamingo-3B-Instruct
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from PIL import Image
import argparse
from scipy import stats


# =============================================================================
# STYLING
# =============================================================================

NORD = {
    "blue": "#5E81AC",
    "red": "#BF616A", 
    "green": "#A3BE8C",
    "sand": "#EBCB8B",
    "ice": "#81A1C1",
    "dark": "#2E3440",
    "light_gray": "#D8DEE9",
    "frost": "#88C0D0",
    "purple": "#B48EAD",
    "orange": "#D08770",
}

FEATURE_PAIRS = {
    'red_green': {
        'feature_a': 'red',
        'feature_b': 'green',
        'color_a': NORD['red'],
        'color_b': NORD['green'],
        'display_name': 'Red / Green',
    },
    'striped_solid': {
        'feature_a': 'striped',
        'feature_b': 'solid',
        'color_a': NORD['ice'],
        'color_b': NORD['dark'],
        'display_name': 'Striped / Solid',
    },
    'one_many': {
        'feature_a': 'one',
        'feature_b': 'many',
        'color_a': NORD['frost'],
        'color_b': NORD['red'],
        'display_name': 'One / Many',
    },
    'horizontal_vertical': {
        'feature_a': 'horizontal',
        'feature_b': 'vertical',
        'color_a': NORD['green'],
        'color_b': NORD['purple'],
        'display_name': 'Horiz. / Vert.',
    },
    'square_circle': {
        'feature_a': 'square',
        'feature_b': 'circle',
        'color_a': NORD['blue'],
        'color_b': NORD['sand'],
        'display_name': 'Square / Circle',
    },
    'empty_filled': {
        'feature_a': 'empty',
        'feature_b': 'filled',
        'color_a': NORD['frost'],   # Light blue for empty
        'color_b': NORD['purple'],  # Purple for filled
        'display_name': 'Empty / Filled',
    },
    'left_right': {
        'feature_a': 'left',
        'feature_b': 'right',
        'color_a': NORD['purple'],
        'color_b': NORD['orange'],
        'display_name': 'Left / Right',
    },
    'top_bottom': {
        'feature_a': 'top',
        'feature_b': 'bottom',
        'color_a': NORD['frost'],
        'color_b': NORD['purple'],
        'display_name': 'Top / Bottom',
    },
}

# Features to exclude from bar plot
EXCLUDE_FROM_BARPLOT = ['top_bottom', 'left_right']


# =============================================================================
# DATA LOADING
# =============================================================================

def load_results(results_dir, layer='layer1'):
    """Load results CSV."""
    results_dir = Path(results_dir)
    df = pd.read_csv(results_dir / f'all_results_{layer}.csv')
    return df


def load_test_images(data_dir, feature_pair, n_images=4):
    """Load actual test images from the synthetic dataset."""
    data_dir = Path(data_dir)
    test_dir = data_dir / f'synthetic_data_{feature_pair}' / 'test'
    
    if not test_dir.exists():
        return None
    
    # Get images where both features are present (has_a=True, has_b=True)
    metadata_path = data_dir / f'synthetic_data_{feature_pair}' / 'test_metadata.csv'
    if metadata_path.exists():
        meta_df = pd.read_csv(metadata_path)
        both_present = meta_df[(meta_df['has_a'] == True) & (meta_df['has_b'] == True)]
        if len(both_present) >= n_images:
            filenames = both_present['filename'].head(n_images).tolist()
        else:
            filenames = [f'test_{i:04d}.png' for i in range(n_images)]
    else:
        filenames = [f'test_{i:04d}.png' for i in range(n_images)]
    
    images = []
    for fname in filenames:
        img_path = test_dir / fname
        if img_path.exists():
            images.append(np.array(Image.open(img_path)))
    
    return images if len(images) == n_images else None


def load_test_images_varied(data_dir, feature_pair):
    """Load 4 test images showing different feature combinations.
    
    Returns images in order: [both, just_a, just_b, both]
    This shows the variety of the dataset.
    """
    data_dir = Path(data_dir)
    test_dir = data_dir / f'synthetic_data_{feature_pair}' / 'test'
    
    if not test_dir.exists():
        return None
    
    metadata_path = data_dir / f'synthetic_data_{feature_pair}' / 'test_metadata.csv'
    if not metadata_path.exists():
        return None
    
    meta_df = pd.read_csv(metadata_path)
    
    images = []
    
    # Get one image for each combination
    combinations = [
        (True, True),   # both features
        (True, False),  # just feature A
        (False, True),  # just feature B  
        (True, True),   # both features (different one)
    ]
    
    for i, (has_a, has_b) in enumerate(combinations):
        subset = meta_df[(meta_df['has_a'] == has_a) & (meta_df['has_b'] == has_b)]
        if len(subset) > i:  # Use different images for the two "both" cases
            fname = subset.iloc[i % len(subset)]['filename']
        elif len(subset) > 0:
            fname = subset.iloc[0]['filename']
        else:
            # Fallback
            fname = f'test_{i:04d}.png'
        
        img_path = test_dir / fname
        if img_path.exists():
            images.append(np.array(Image.open(img_path)))
        else:
            return None
    
    return images if len(images) == 4 else None


# =============================================================================
# PLOTTING COMPONENTS
# =============================================================================

def plot_image_grid(axes, images):
    """Plot images in a 2x2 grid of axes."""
    for ax, img in zip(axes.flat, images):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color(NORD['light_gray'])


def plot_scatter(ax, df, feature_pair):
    """Plot VCR vs Interventional Effect scatter for a feature pair."""
    pair_info = FEATURE_PAIRS[feature_pair]
    fp_data = df[df['feature_pair'] == feature_pair]
    
    if len(fp_data) == 0:
        ax.text(0.5, 0.5, f'No data', ha='center', va='center', fontsize=10)
        return
    
    for feat in [pair_info['feature_a'], pair_info['feature_b']]:
        feat_data = fp_data[fp_data['feature'] == feat]
        color = pair_info['color_a'] if feat == pair_info['feature_a'] else pair_info['color_b']
        ax.scatter(feat_data['vcr_sensitivity'], feat_data['interventional_effect'],
                   alpha=0.7, s=45, c=color, label=feat, edgecolor='white', linewidth=0.5)
    
    # Regression line
    if len(fp_data) > 2:
        slope, intercept, r, p, _ = stats.linregress(
            fp_data['vcr_sensitivity'], fp_data['interventional_effect']
        )
        x_line = np.linspace(fp_data['vcr_sensitivity'].min(), 
                            fp_data['vcr_sensitivity'].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, '--', color=NORD['dark'], 
                linewidth=2, alpha=0.7)
        
        # Add r value
        ax.text(0.95, 0.05, f'r = {r:.2f}', transform=ax.transAxes, 
                ha='right', va='bottom', fontsize=11, fontweight='bold',
                color=NORD['dark'])
    
    ax.axhline(0, color=NORD['light_gray'], linestyle='-', linewidth=1)
    ax.axvline(0, color=NORD['light_gray'], linestyle='-', linewidth=1)
    
    ax.set_xlabel('VCR Sensitivity', fontsize=10, color=NORD['dark'])
    ax.set_ylabel('Interventional Effect', fontsize=10, color=NORD['dark'])
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(NORD['dark'])
    ax.spines['bottom'].set_color(NORD['dark'])
    ax.tick_params(colors=NORD['dark'])


def plot_correlation_bars(ax, df, exclude_pairs=None):
    """Plot horizontal bar chart of correlations with rounded bars."""
    if exclude_pairs is None:
        exclude_pairs = EXCLUDE_FROM_BARPLOT
    
    # Calculate correlations for each feature pair
    correlations = []
    feature_pairs = [fp for fp in df['feature_pair'].unique() if fp not in exclude_pairs]
    
    for fp in feature_pairs:
        fp_data = df[df['feature_pair'] == fp]
        if len(fp_data) > 2:
            r, p = stats.pearsonr(fp_data['vcr_sensitivity'], fp_data['interventional_effect'])
            correlations.append({
                'feature_pair': fp,
                'display_name': FEATURE_PAIRS[fp]['display_name'],
                'r': r,
                'p': p
            })
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('r', ascending=True)
    
    y_pos = np.arange(len(corr_df))
    
    # Draw rounded bars
    bar_height = 0.5
    for i, (_, row) in enumerate(corr_df.iterrows()):
        color = NORD['blue'] if row['r'] > 0 else NORD['red']
        
        width = abs(row['r'])
        x_start = 0 if row['r'] >= 0 else row['r']
        
        rect = FancyBboxPatch(
            (x_start, i - bar_height/2), width, bar_height,
            boxstyle="round,pad=0,rounding_size=0.06",
            facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.85
        )
        ax.add_patch(rect)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(corr_df['display_name'], fontsize=10)
    ax.set_xlabel('Pearson r', fontsize=10, color=NORD['dark'])
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.5, len(corr_df) - 0.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(NORD['dark'])
    ax.spines['bottom'].set_linewidth(0.5)
    ax.tick_params(left=False, colors=NORD['dark'])


def plot_concordance_bars(ax, df, exclude_pairs=None):
    """Plot horizontal bar chart of concordance percentages with rounded bars."""
    if exclude_pairs is None:
        exclude_pairs = EXCLUDE_FROM_BARPLOT
    
    # Calculate concordance for each feature pair
    concordances = []
    feature_pairs = [fp for fp in df['feature_pair'].unique() if fp not in exclude_pairs]
    
    for fp in feature_pairs:
        fp_data = df[df['feature_pair'] == fp]
        if len(fp_data) > 2:
            # Calculate concordance (sign agreement)
            vcr_signs = np.sign(fp_data['vcr_sensitivity'].values)
            int_signs = np.sign(fp_data['interventional_effect'].values)
            concordance = np.mean(vcr_signs == int_signs) * 100
            
            # Also get r for sorting consistency
            r, _ = stats.pearsonr(fp_data['vcr_sensitivity'], fp_data['interventional_effect'])
            concordances.append({
                'feature_pair': fp,
                'display_name': FEATURE_PAIRS[fp]['display_name'],
                'concordance': concordance,
                'r': r
            })
    
    conc_df = pd.DataFrame(concordances)
    conc_df = conc_df.sort_values('r', ascending=True)  # Same order as correlation plot
    
    y_pos = np.arange(len(conc_df))
    
    # Draw rounded bars
    bar_height = 0.5
    for i, (_, row) in enumerate(conc_df.iterrows()):
        color = NORD['green']
        
        width = row['concordance'] / 100  # Convert to 0-1 scale
        
        rect = FancyBboxPatch(
            (0, i - bar_height/2), width, bar_height,
            boxstyle="round,pad=0,rounding_size=0.06",
            facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.85
        )
        ax.add_patch(rect)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(conc_df['display_name'], fontsize=10)
    ax.set_xlabel('Concordance (%)', fontsize=10, color=NORD['dark'])
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.5, len(conc_df) - 0.5)
    
    # Add percentage labels on x-axis
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0', '25', '50', '75', '100'])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(NORD['dark'])
    ax.spines['bottom'].set_linewidth(0.5)
    ax.tick_params(left=False, colors=NORD['dark'])


# =============================================================================
# VCR VS CLIP COMPARISON PLOTS
# =============================================================================

def plot_vcr_vs_clip_scatter(df, output_dir, layer_label='-1', exclude_pairs=None):
    """Generate side-by-side scatter comparison between VCR and CLIP-only baseline."""
    output_dir = Path(output_dir)
    
    if exclude_pairs is None:
        exclude_pairs = EXCLUDE_FROM_BARPLOT
    
    # Check if clip_score column exists
    if 'clip_score' not in df.columns:
        print(f"Warning: 'clip_score' column not found in data. Skipping VCR vs CLIP scatter plot.")
        return None, None
    
    # Filter out excluded feature pairs
    df = df[~df['feature_pair'].isin(exclude_pairs)]
    
    feature_pairs = df['feature_pair'].unique()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    
    # VCR scatter
    ax = axes[0]
    for fp in feature_pairs:
        fp_data = df[df['feature_pair'] == fp]
        pair_info = FEATURE_PAIRS[fp]
        for feat in [pair_info['feature_a'], pair_info['feature_b']]:
            feat_data = fp_data[fp_data['feature'] == feat]
            color = pair_info['color_a'] if feat == pair_info['feature_a'] else pair_info['color_b']
            ax.scatter(feat_data['vcr_sensitivity'], feat_data['interventional_effect'],
                      alpha=0.5, s=30, c=color, edgecolor='white', linewidth=0.3)
    
    slope, intercept, r_vcr, p_vcr, _ = stats.linregress(
        df['vcr_sensitivity'], df['interventional_effect']
    )
    x_line = np.linspace(df['vcr_sensitivity'].min(), df['vcr_sensitivity'].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=2)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('VCR Sensitivity', fontsize=11, color=NORD['dark'])
    ax.set_ylabel('Interventional Effect', fontsize=11, color=NORD['dark'])
    ax.set_title(f'VCR (gradient-based)\nr={r_vcr:.3f}, $R^2$={r_vcr**2:.3f}', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # CLIP scatter
    ax = axes[1]
    for fp in feature_pairs:
        fp_data = df[df['feature_pair'] == fp]
        pair_info = FEATURE_PAIRS[fp]
        for feat in [pair_info['feature_a'], pair_info['feature_b']]:
            feat_data = fp_data[fp_data['feature'] == feat]
            color = pair_info['color_a'] if feat == pair_info['feature_a'] else pair_info['color_b']
            ax.scatter(feat_data['clip_score'], feat_data['interventional_effect'],
                      alpha=0.5, s=30, c=color, edgecolor='white', linewidth=0.3)
    
    slope, intercept, r_clip, p_clip, _ = stats.linregress(
        df['clip_score'], df['interventional_effect']
    )
    x_line = np.linspace(df['clip_score'].min(), df['clip_score'].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=2)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('CLIP Correlation', fontsize=11, color=NORD['dark'])
    ax.set_ylabel('Interventional Effect', fontsize=11, color=NORD['dark'])
    ax.set_title(f'CLIP-only (no gradients)\nr={r_clip:.3f}, $R^2$={r_clip**2:.3f}', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.suptitle(f'VCR vs CLIP-only Baseline (Layer {layer_label})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f'vcr_vs_clip_scatter_layer{layer_label}.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir / f'vcr_vs_clip_scatter_layer{layer_label}.png'}")
    plt.close()
    
    return r_vcr, r_clip


def plot_vcr_vs_clip_by_feature(df, output_dir, layer_label='-1', exclude_pairs=None):
    """Generate per-feature-pair correlation comparison bar chart (VCR vs CLIP)."""
    output_dir = Path(output_dir)
    
    if exclude_pairs is None:
        exclude_pairs = EXCLUDE_FROM_BARPLOT
    
    # Check if clip_score column exists
    if 'clip_score' not in df.columns:
        print(f"Warning: 'clip_score' column not found in data. Skipping VCR vs CLIP by feature plot.")
        return None
    
    # Filter to non-excluded feature pairs
    feature_pairs = [fp for fp in df['feature_pair'].unique() if fp not in exclude_pairs]
    
    correlations = []
    for fp in feature_pairs:
        fp_data = df[df['feature_pair'] == fp]
        if len(fp_data) > 2:
            r_vcr_fp, p_vcr_fp = stats.pearsonr(fp_data['vcr_sensitivity'], fp_data['interventional_effect'])
            r_clip_fp, p_clip_fp = stats.pearsonr(fp_data['clip_score'], fp_data['interventional_effect'])
            correlations.append({
                'feature_pair': fp,
                'display_name': FEATURE_PAIRS[fp]['display_name'],
                'method': 'VCR',
                'r': r_vcr_fp,
                'p': p_vcr_fp
            })
            correlations.append({
                'feature_pair': fp,
                'display_name': FEATURE_PAIRS[fp]['display_name'],
                'method': 'CLIP',
                'r': r_clip_fp,
                'p': p_clip_fp
            })
    
    corr_df = pd.DataFrame(correlations)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    
    x = np.arange(len(feature_pairs))
    width = 0.35
    
    vcr_r = corr_df[corr_df['method'] == 'VCR']['r'].values
    clip_r = corr_df[corr_df['method'] == 'CLIP']['r'].values
    display_names = corr_df[corr_df['method'] == 'VCR']['display_name'].values
    
    bars1 = ax.bar(x - width/2, vcr_r, width, label='VCR (gradient-based)', color=NORD['blue'], alpha=0.8)
    bars2 = ax.bar(x + width/2, clip_r, width, label='CLIP-only', color=NORD['orange'], alpha=0.8)
    
    ax.set_ylabel('Pearson r', fontsize=11, color=NORD['dark'])
    ax.set_title(f'Correlation with Interventional Effects: VCR vs CLIP-only (Layer {layer_label})', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=10)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_ylim(-1, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add significance markers
    for i, (r1, r2) in enumerate(zip(vcr_r, clip_r)):
        p1 = corr_df[(corr_df['method'] == 'VCR') & (corr_df['feature_pair'] == feature_pairs[i])]['p'].values[0]
        p2 = corr_df[(corr_df['method'] == 'CLIP') & (corr_df['feature_pair'] == feature_pairs[i])]['p'].values[0]
        
        sig1 = '***' if p1 < 0.001 else '**' if p1 < 0.01 else '*' if p1 < 0.05 else ''
        sig2 = '***' if p2 < 0.001 else '**' if p2 < 0.01 else '*' if p2 < 0.05 else ''
        
        y_offset = 0.03 if r1 >= 0 else -0.08
        ax.text(i - width/2, r1 + y_offset, sig1, ha='center', fontsize=8)
        y_offset = 0.03 if r2 >= 0 else -0.08
        ax.text(i + width/2, r2 + y_offset, sig2, ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'vcr_vs_clip_by_feature_layer{layer_label}.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir / f'vcr_vs_clip_by_feature_layer{layer_label}.png'}")
    plt.close()
    
    return corr_df


def plot_vcr_vs_clip_overall(df, output_dir, layer_label='-1', exclude_pairs=None):
    """Generate overall summary bar chart comparing VCR and CLIP."""
    output_dir = Path(output_dir)
    
    if exclude_pairs is None:
        exclude_pairs = EXCLUDE_FROM_BARPLOT
    
    # Check if clip_score column exists
    if 'clip_score' not in df.columns:
        print(f"Warning: 'clip_score' column not found in data. Skipping VCR vs CLIP overall plot.")
        return None, None
    
    # Filter out excluded feature pairs
    df = df[~df['feature_pair'].isin(exclude_pairs)]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    
    r_vcr_all, p_vcr_all = stats.pearsonr(df['vcr_sensitivity'], df['interventional_effect'])
    r_clip_all, p_clip_all = stats.pearsonr(df['clip_score'], df['interventional_effect'])
    
    methods = ['VCR\n(gradient-based)', 'CLIP-only']
    r_values = [r_vcr_all, r_clip_all]
    r2_values = [r_vcr_all**2, r_clip_all**2]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, r_values, width, label='Pearson r', color=NORD['blue'], alpha=0.8)
    bars2 = ax.bar(x + width/2, r2_values, width, label='$R^2$', color=NORD['green'], alpha=0.8)
    
    ax.set_ylabel('Correlation', fontsize=11, color=NORD['dark'])
    ax.set_title(f'Overall Correlation with Interventional Effects (Layer {layer_label})', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels
    for bar, val in zip(bars1, r_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', 
                ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, r2_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', 
                ha='center', va='bottom', fontsize=10)
    
    # Add p-value annotations
    ax.text(0, -0.1, f'p={p_vcr_all:.2e}', ha='center', fontsize=9, transform=ax.get_xaxis_transform())
    ax.text(1, -0.1, f'p={p_clip_all:.2e}', ha='center', fontsize=9, transform=ax.get_xaxis_transform())
    
    plt.tight_layout()
    plt.savefig(output_dir / f'vcr_vs_clip_overall_layer{layer_label}.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir / f'vcr_vs_clip_overall_layer{layer_label}.png'}")
    plt.close()
    
    return r_vcr_all, r_clip_all


def plot_vcr_vs_clip_comparison(df, output_dir, layer_label='-1', exclude_pairs=None):
    """Generate all VCR vs CLIP comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if exclude_pairs is None:
        exclude_pairs = EXCLUDE_FROM_BARPLOT
    
    # Check if clip_score column exists
    if 'clip_score' not in df.columns:
        print(f"Warning: 'clip_score' column not found in data. Skipping all VCR vs CLIP plots.")
        return None, None, None
    
    # Plot 1: Side-by-side scatter
    r_vcr, r_clip = plot_vcr_vs_clip_scatter(df, output_dir, layer_label, exclude_pairs)
    
    # Plot 2: Per-feature-pair bar chart
    corr_df = plot_vcr_vs_clip_by_feature(df, output_dir, layer_label, exclude_pairs)
    
    # Plot 3: Overall summary
    r_vcr_overall, r_clip_overall = plot_vcr_vs_clip_overall(df, output_dir, layer_label, exclude_pairs)
    
    return corr_df, r_vcr_overall, r_clip_overall


# =============================================================================
# MAIN COMPOSITE FIGURE
# =============================================================================

def create_examples_figure(df, output_dir, layer_label='-1', data_dir=None):
    """Create figure with example images and scatter plots."""
    output_dir = Path(output_dir)
    if data_dir is None:
        data_dir = output_dir.parent
    
    fig = plt.figure(figsize=(7, 6))
    fig.patch.set_facecolor('white')
    
    # Two rows - more vertical space between them
    gs_main = gridspec.GridSpec(2, 1, hspace=0.45,
                                 left=0.02, right=0.98, top=0.95, bottom=0.08)
    
    # --- Row 1: Square/Circle ---
    gs_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0], 
                                             width_ratios=[0.38, 0.62], wspace=0.22)
    
    # 2x2 image grid
    gs_a_imgs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_a[0], 
                                                   hspace=0.02, wspace=0.02)
    axes_a_imgs = np.array([[fig.add_subplot(gs_a_imgs[i, j]) for j in range(2)] for i in range(2)])
    
    images_a = load_test_images_varied(data_dir, 'square_circle')
    if images_a:
        plot_image_grid(axes_a_imgs, images_a)
    else:
        for ax in axes_a_imgs.flat:
            ax.text(0.5, 0.5, 'No image', ha='center', va='center', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Scatter plot
    ax_a_scatter = fig.add_subplot(gs_a[1])
    plot_scatter(ax_a_scatter, df, 'square_circle')
    
    # Title - further up and to the right
    fig.text(0.55, 0.995, 'Square / Circle', fontsize=11, fontweight='bold', ha='center', va='top')
    
    # --- Row 2: Empty/Filled ---
    gs_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[1],
                                             width_ratios=[0.38, 0.62], wspace=0.22)
    
    # 2x2 image grid
    gs_b_imgs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_b[0],
                                                   hspace=0.02, wspace=0.02)
    axes_b_imgs = np.array([[fig.add_subplot(gs_b_imgs[i, j]) for j in range(2)] for i in range(2)])
    
    images_b = load_test_images_varied(data_dir, 'empty_filled')
    if images_b:
        plot_image_grid(axes_b_imgs, images_b)
    else:
        for ax in axes_b_imgs.flat:
            ax.text(0.5, 0.5, 'No image', ha='center', va='center', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Scatter plot
    ax_b_scatter = fig.add_subplot(gs_b[1])
    plot_scatter(ax_b_scatter, df, 'empty_filled')
    
    # Title - further up and to the right
    fig.text(0.55, 0.49, 'Empty / Filled', fontsize=11, fontweight='bold', ha='center', va='top')
    
    plt.savefig(output_dir / f'synthetic_examples_layer{layer_label}.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir / f'synthetic_examples_layer{layer_label}.png'}")
    plt.close()


def create_summary_figure(df, output_dir, layer_label='-1'):
    """Create figure with correlation bar chart summary."""
    output_dir = Path(output_dir)
    
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('white')
    
    plot_correlation_bars(ax, df)
    ax.set_title('VCR–Intervention Correlation\nby Feature Pair', fontsize=11, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'synthetic_summary_layer{layer_label}.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir / f'synthetic_summary_layer{layer_label}.png'}")
    plt.close()


def create_composite_figure(df, output_dir, layer_label='-1', data_dir=None):
    """Create all figures including VCR vs CLIP comparison."""
    create_examples_figure(df, output_dir, layer_label, data_dir)
    create_summary_figure(df, output_dir, layer_label)
    
    # Add VCR vs CLIP comparison plots
    vcr_clip_corr_df, r_vcr, r_clip = plot_vcr_vs_clip_comparison(df, output_dir, layer_label)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing all_results_layer1.csv and all_results_layer4.csv')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing synthetic_data_* folders (defaults to results_dir parent)')
    parser.add_argument('--layer', type=str, default='1', choices=['1', '4', 'both'],
                        help='Which layer(s) to plot')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir) if args.data_dir else results_dir.parent
    
    print(f"Loading results from {results_dir}")
    print(f"Looking for images in {data_dir}")
    
    if args.layer in ['1', 'both']:
        df_layer1 = load_results(results_dir, 'layer1')
        print(f"Layer 1: {len(df_layer1)} rows")
        print(f"Columns: {list(df_layer1.columns)}")
        create_composite_figure(df_layer1, results_dir, '-1', data_dir)
    
    if args.layer in ['4', 'both']:
        df_layer4 = load_results(results_dir, 'layer4')
        print(f"Layer 4: {len(df_layer4)} rows")
        print(f"Columns: {list(df_layer4.columns)}")
        create_composite_figure(df_layer4, results_dir, '-4', data_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()