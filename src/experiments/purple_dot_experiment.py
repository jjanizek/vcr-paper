import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as T
import random
from models.flamingo import FlamingoAPI

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

def set_nord_style():
    """Apply a Nord-inspired scientific style globally."""
    nord_colors = [
        "#5E81AC",  # steel blue
        "#BF616A",  # faded red
        "#A3BE8C",  # pale green
        "#EBCB8B",  # sand
        "#81A1C1",  # ice blue
    ]
    sns.set_theme(
        style="white",
        rc={
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#2E3440",
            "axes.labelcolor": "#2E3440",
            "xtick.color": "#2E3440",
            "ytick.color": "#2E3440",
            "text.color": "#2E3440",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#D8DEE9",
            "grid.alpha": 0.6,
            "grid.linewidth": 0.6,
            "grid.linestyle": ":",
            "lines.linewidth": 2.0,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.dpi": 200,
        },
    )
    sns.set_palette(nord_colors)


def add_purple_dots(image: Image.Image, num_dots: int = 4, dot_radius: int = 30, color: tuple = (128, 0, 128)) -> Image.Image:
    """Add small colored dots to an image at random locations."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    for _ in range(num_dots):
        x = random.randint(dot_radius, w - dot_radius)
        y = random.randint(dot_radius, h - dot_radius)
        bbox = [x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius]
        draw.ellipse(bbox, fill=color)
    
    return img


# Define colors for each skin type
DOT_COLOR_FST12 = (60, 34, 112)   # #3C2270 - blue-purple
DOT_COLOR_FST56 = (60, 34, 112)   # Same color for consistency


def get_augmentation_transform(max_rotation: float = 15):
    """Return augmentation transform with rotation, shifts, crops."""
    return T.Compose([
        T.RandomRotation(degrees=max_rotation),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        T.RandomResizedCrop(size=(224, 224), scale=(0.85, 1.0)),
    ])


def load_images_from_folder(folder_path: str) -> list[tuple[str, Image.Image]]:
    """Load all images from a folder."""
    images = []
    folder = Path(folder_path)
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        for img_path in folder.glob(ext):
            img = Image.open(img_path).convert('RGB')
            images.append((str(img_path), img))
    return images


def get_malignant_logprob(api: FlamingoAPI, prompt: str, image_paths: list[str]) -> float:
    """Get the log probability of 'Malignant' output."""
    choices = ["Benign", "Malignant"]
    _, logprobs = api.get_best_choice(prompt, choices, image_paths)
    return logprobs["Malignant"]


def run_condition(
    api: FlamingoAPI,
    images: list[tuple[str, Image.Image]],
    add_dots: bool,
    use_icl: bool,
    demo_examples: list[tuple[str, str]] = None,
    augment: bool = True,
    num_augments: int = 10,
    dot_color: tuple = (128, 0, 128),
    temp_dir: str = "/tmp/purple_dot_exp"
) -> list[float]:
    """Run experiment for a single condition, return list of Malignant logprobs."""
    os.makedirs(temp_dir, exist_ok=True)
    
    aug_transform = get_augmentation_transform() if augment else None
    logprobs = []
    
    # Calculate total iterations for progress bar
    n_copies = num_augments if augment else 1
    total_iters = len(images) * n_copies
    
    with tqdm(total=total_iters, desc=f"dots={add_dots}, icl={use_icl}") as pbar:
        for img_path, img in images:
            for aug_idx in range(n_copies):
                # Process image
                processed = img.copy()
                if augment and aug_transform:
                    processed = aug_transform(processed)
                if add_dots:
                    processed = add_purple_dots(processed, color=dot_color)
                
                # Save processed image temporarily
                test_img_path = os.path.join(temp_dir, "test_img.png")
                processed.save(test_img_path)
                
                # Build prompt and image paths
                if use_icl and demo_examples:
                    prompt = (
                        "You are a medical image analysis assistant. For each skin lesion image, "
                        "choose between Benign and Malignant.\n\nExamples:\n\n"
                    )
                    image_paths = []
                    
                    # Add demo images with real labels
                    for demo_path, demo_label in demo_examples:
                        image_paths.append(demo_path)
                        prompt += f"<image>\nThe lesion is {demo_label}.\n\n"
                    
                    image_paths.append(test_img_path)
                    prompt += "<image>\nThe lesion is"
                else:
                    # Zero-shot
                    prompt = (
                        "You are a medical image analysis assistant. For each skin lesion image, "
                        "choose between Benign and Malignant.\n\n"
                        "<image>\nThe lesion is"
                    )
                    image_paths = [test_img_path]
                
                lp = get_malignant_logprob(api, prompt, image_paths)
                logprobs.append(lp)
                pbar.update(1)
    
    return logprobs


def draw_rounded_bar(ax, x, y, width, height, radius=0.08, color=NORD["blue"], alpha=1.0):
    """Draw a bar with rounded corners using FancyBboxPatch."""
    if height >= 0:
        bottom = 0
        bar_height = height
    else:
        bottom = height
        bar_height = abs(height)
    
    # Create rounded rectangle
    fancy_box = FancyBboxPatch(
        (x - width/2, bottom),
        width,
        bar_height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color,
        edgecolor='none',
        alpha=alpha,
        zorder=3
    )
    ax.add_patch(fancy_box)
    return fancy_box


def plot_results_with_examples(
    fst12_images: list[tuple[str, Image.Image]],
    fst56_images: list[tuple[str, Image.Image]],
    diffs: dict,
    ses: dict,
    output_path: str,
    seed: int = 42
):
    """
    Create a figure with example images on the left and bar plots on the right.
    Layout: 2 rows (FST I/II top, FST V/VI bottom)
    Each row: [Original image] [Image + dots] [Bar plot showing effect]
    """
    set_nord_style()
    
    # Set seed for reproducible dot placement
    random.seed(seed)
    
    # Create figure with custom grid
    fig = plt.figure(figsize=(14, 8))
    
    # Define grid: 2 rows, with images taking ~40% and bar plots ~60%
    gs = fig.add_gridspec(
        2, 4, 
        width_ratios=[1, 1, 0.1, 1.8],
        height_ratios=[1, 1],
        hspace=0.3,
        wspace=0.15
    )
    
    # Pick example images
    fst12_path, fst12_img = fst12_images[0]
    fst56_path, fst56_img = fst56_images[0]
    
    # Create images with dots
    random.seed(seed)  # Reset for consistent dot placement
    fst12_dots = add_purple_dots(fst12_img.copy(), color=DOT_COLOR_FST12)
    random.seed(seed + 1)
    fst56_dots = add_purple_dots(fst56_img.copy(), color=DOT_COLOR_FST56)
    
    # --- Row 1: FST I/II ---
    # Original image
    ax_fst12_orig = fig.add_subplot(gs[0, 0])
    ax_fst12_orig.imshow(fst12_img)
    ax_fst12_orig.set_title("FST I/II", fontsize=11, fontweight='bold', color=NORD["dark"])
    ax_fst12_orig.axis('off')
    
    # Image with dots
    ax_fst12_dots = fig.add_subplot(gs[0, 1])
    ax_fst12_dots.imshow(fst12_dots)
    ax_fst12_dots.set_title("FST I/II + purple dots", fontsize=11, fontweight='bold', color=NORD["dark"])
    ax_fst12_dots.axis('off')
    
    # Bar plot for FST I/II
    ax_fst12_bar = fig.add_subplot(gs[0, 3])
    
    # Draw rounded bars
    bar_width = 0.5
    x_positions = [0, 1]
    y_values = [diffs["FST12_ZeroShot"], diffs["FST12_ICL"]]
    y_errors = [ses["FST12_ZeroShot"], ses["FST12_ICL"]]
    
    for i, (x, y, err) in enumerate(zip(x_positions, y_values, y_errors)):
        draw_rounded_bar(ax_fst12_bar, x, 0, bar_width, y, radius=0.06, color=NORD["blue"])
        # Add error bars
        ax_fst12_bar.errorbar(x, y, yerr=err, fmt='none', color=NORD["dark"], 
                              capsize=4, capthick=1.5, elinewidth=1.5, zorder=4)
    
    ax_fst12_bar.set_xlim(-0.7, 1.7)
    ax_fst12_bar.set_xticks(x_positions)
    ax_fst12_bar.set_xticklabels(["Zero-Shot", "ICL"], fontsize=10)
    ax_fst12_bar.set_ylabel("Δ logprob(Malignant)", fontsize=10, color=NORD["dark"])
    ax_fst12_bar.axhline(0, color=NORD["light_gray"], linestyle='--', linewidth=1.2, zorder=1)
    ax_fst12_bar.set_title("Effect of Purple Dots", fontsize=11, fontweight='bold', color=NORD["dark"])
    ax_fst12_bar.spines['top'].set_visible(False)
    ax_fst12_bar.spines['right'].set_visible(False)
    ax_fst12_bar.spines['left'].set_color(NORD["dark"])
    ax_fst12_bar.spines['bottom'].set_color(NORD["dark"])
    
    # --- Row 2: FST V/VI ---
    # Original image
    ax_fst56_orig = fig.add_subplot(gs[1, 0])
    ax_fst56_orig.imshow(fst56_img)
    ax_fst56_orig.set_title("FST V/VI", fontsize=11, fontweight='bold', color=NORD["dark"])
    ax_fst56_orig.axis('off')
    
    # Image with dots
    ax_fst56_dots = fig.add_subplot(gs[1, 1])
    ax_fst56_dots.imshow(fst56_dots)
    ax_fst56_dots.set_title("FST V/VI + purple dots", fontsize=11, fontweight='bold', color=NORD["dark"])
    ax_fst56_dots.axis('off')
    
    # Bar plot for FST V/VI
    ax_fst56_bar = fig.add_subplot(gs[1, 3])
    
    y_values = [diffs["FST56_ZeroShot"], diffs["FST56_ICL"]]
    y_errors = [ses["FST56_ZeroShot"], ses["FST56_ICL"]]
    
    for i, (x, y, err) in enumerate(zip(x_positions, y_values, y_errors)):
        draw_rounded_bar(ax_fst56_bar, x, 0, bar_width, y, radius=0.06, color=NORD["red"])
        # Add error bars
        ax_fst56_bar.errorbar(x, y, yerr=err, fmt='none', color=NORD["dark"], 
                              capsize=4, capthick=1.5, elinewidth=1.5, zorder=4)
    
    ax_fst56_bar.set_xlim(-0.7, 1.7)
    ax_fst56_bar.set_xticks(x_positions)
    ax_fst56_bar.set_xticklabels(["Zero-Shot", "ICL"], fontsize=10)
    ax_fst56_bar.set_ylabel("Δ logprob(Malignant)", fontsize=10, color=NORD["dark"])
    ax_fst56_bar.axhline(0, color=NORD["light_gray"], linestyle='--', linewidth=1.2, zorder=1)
    ax_fst56_bar.set_title("Effect of Purple Dots", fontsize=11, fontweight='bold', color=NORD["dark"])
    ax_fst56_bar.spines['top'].set_visible(False)
    ax_fst56_bar.spines['right'].set_visible(False)
    ax_fst56_bar.spines['left'].set_color(NORD["dark"])
    ax_fst56_bar.spines['bottom'].set_color(NORD["dark"])
    
    # Match y-axis limits across both bar plots
    all_values = [diffs["FST12_ZeroShot"], diffs["FST12_ICL"], 
                  diffs["FST56_ZeroShot"], diffs["FST56_ICL"]]
    all_errors = [ses["FST12_ZeroShot"], ses["FST12_ICL"], 
                  ses["FST56_ZeroShot"], ses["FST56_ICL"]]
    y_max = max([v + e for v, e in zip(all_values, all_errors)]) * 1.2
    y_min = min([v - e for v, e in zip(all_values, all_errors)]) * 1.2
    
    # Ensure zero is visible
    y_max = max(y_max, 0.1)
    y_min = min(y_min, -0.1)
    
    ax_fst12_bar.set_ylim(y_min, y_max)
    ax_fst56_bar.set_ylim(y_min, y_max)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=NORD["blue"], label='FST I/II'),
        mpatches.Patch(facecolor=NORD["red"], label='FST V/VI')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98),
               frameon=False, fontsize=10)
    
    # Main title
    fig.suptitle("Effect of Purple Dots on Malignancy Prediction", 
                 fontsize=14, fontweight='bold', color=NORD["dark"], y=0.98)
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to {output_path}")
    plt.close()


def save_example_dots_plot(
    fst12_images: list[tuple[str, Image.Image]],
    fst56_images: list[tuple[str, Image.Image]],
    output_path: str,
    seed: int = 42
):
    """Save a plot showing example images before and after adding colored dots (legacy function)."""
    set_nord_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
    # Pick one example from each skin type
    fst12_path, fst12_img = fst12_images[0]
    fst56_path, fst56_img = fst56_images[0]
    
    # FST12 original
    axes[0, 0].imshow(fst12_img)
    axes[0, 0].set_title("FST I/II", fontsize=11, fontweight='bold', color=NORD["dark"])
    axes[0, 0].axis('off')
    
    # FST12 with dots (blue-purple)
    random.seed(seed)
    fst12_dots = add_purple_dots(fst12_img.copy(), color=DOT_COLOR_FST12)
    axes[0, 1].imshow(fst12_dots)
    axes[0, 1].set_title("FST I/II + purple dots", fontsize=11, fontweight='bold', color=NORD["dark"])
    axes[0, 1].axis('off')
    
    # FST56 original
    axes[1, 0].imshow(fst56_img)
    axes[1, 0].set_title("FST V/VI", fontsize=11, fontweight='bold', color=NORD["dark"])
    axes[1, 0].axis('off')
    
    # FST56 with dots
    random.seed(seed + 1)
    fst56_dots = add_purple_dots(fst56_img.copy(), color=DOT_COLOR_FST56)
    axes[1, 1].imshow(fst56_dots)
    axes[1, 1].set_title("FST V/VI + purple dots", fontsize=11, fontweight='bold', color=NORD["dark"])
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Example dots plot saved to {output_path}")
    plt.close()


def load_demo_examples(
    demo_metadata_path: str,
    base_directory: str,
    num_malignant: int = 4,
    benign_multiplier: int = 3,
    skin_type: str = 'both',
    random_state: int = 42
) -> list[tuple[str, str]]:
    """Load demo examples with real labels from DDI metadata."""
    demo_frame = pd.read_csv(demo_metadata_path, index_col=0)
    num_benign = benign_multiplier * num_malignant
    
    if skin_type == 'both':
        fst12_ben = fst12_mal = num_benign // 2
        fst56_ben = fst56_mal = num_benign // 2
        fst12_mal = fst56_mal = num_malignant // 2
    elif skin_type == 12:
        fst12_ben, fst12_mal = num_benign, num_malignant
        fst56_ben = fst56_mal = 0
    else:  # 56
        fst12_ben = fst12_mal = 0
        fst56_ben, fst56_mal = num_benign, num_malignant
    
    samples = []
    for skin_tone, n_ben, n_mal in [(12, fst12_ben, fst12_mal), (56, fst56_ben, fst56_mal)]:
        subset = demo_frame[demo_frame.skin_tone == skin_tone]
        if n_ben > 0:
            samples.append(subset[subset.malignant == False].sample(n_ben, random_state=random_state))
        if n_mal > 0:
            samples.append(subset[subset.malignant == True].sample(n_mal, random_state=random_state))
    
    final_frame = pd.concat(samples).sample(frac=1, random_state=random_state)
    
    demo_examples = []
    for _, row in final_frame.iterrows():
        label = "Malignant" if row.malignant else "Benign"
        image_path = os.path.join(base_directory, row.DDI_file)
        demo_examples.append((image_path, label))
    
    return demo_examples


def main():
    parser = argparse.ArgumentParser(description="Purple dot experiment for VLM bias analysis")
    parser.add_argument("--fst12_folder", type=str, required=True, help="Path to FST12 blank skin images")
    parser.add_argument("--fst56_folder", type=str, required=True, help="Path to FST56 blank skin images")
    parser.add_argument("--ddi_base_dir", type=str, default="your/path/to/ddi/ddidiversedermatologyimages/",
                        help="Base directory for DDI dataset")
    parser.add_argument("--demo_metadata", type=str, default="your/path/to/ddi_demo_metadata.csv",
                        help="Path to demo metadata CSV")
    parser.add_argument("--num_malignant", type=int, default=4, help="Number of malignant demo images")
    parser.add_argument("--benign_multiplier", type=int, default=3, help="Ratio of benign:malignant demos")
    parser.add_argument("--demo_skin_type", type=str, default='both', help="Skin type for demos: 12, 56, or both")
    parser.add_argument("--num_augments", type=int, default=10, help="Number of augmented copies per image")
    parser.add_argument("--no_augment", action="store_true", help="Disable augmentations")
    parser.add_argument("--output", type=str, default="purple_dot_results.png", help="Output plot path")
    parser.add_argument("--example_output", type=str, default="purple_dot_examples.png", help="Example dots plot path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    augment = not args.no_augment
    
    # Parse demo_skin_type
    demo_skin_type = args.demo_skin_type
    if demo_skin_type.isdigit():
        demo_skin_type = int(demo_skin_type)
    
    print("Loading model...")
    api = FlamingoAPI(model_name='OpenFlamingo-3B-Instruct')
    
    print("Loading images...")
    fst12_images = load_images_from_folder(args.fst12_folder)
    fst56_images = load_images_from_folder(args.fst56_folder)
    print(f"Loaded {len(fst12_images)} FST12 images, {len(fst56_images)} FST56 images")
    
    # Save legacy example dots plot
    save_example_dots_plot(fst12_images, fst56_images, args.example_output, seed=args.seed)
    
    # Load demo examples with real labels from DDI
    demo_examples = load_demo_examples(
        args.demo_metadata,
        args.ddi_base_dir,
        num_malignant=args.num_malignant,
        benign_multiplier=args.benign_multiplier,
        skin_type=demo_skin_type,
        random_state=args.seed
    )
    print(f"Loaded {len(demo_examples)} demo examples")
    
    results = {}
    
    # Run all conditions
    for skin_type, images, dot_color in [("FST12", fst12_images, DOT_COLOR_FST12), ("FST56", fst56_images, DOT_COLOR_FST56)]:
        for use_icl in [False, True]:
            for add_dots in [False, True]:
                key = (skin_type, "ICL" if use_icl else "ZeroShot", "Dots" if add_dots else "NoDots")
                print(f"\nRunning: {key}")
                results[key] = run_condition(
                    api, images, add_dots, use_icl,
                    demo_examples=demo_examples if use_icl else None,
                    augment=augment,
                    num_augments=args.num_augments,
                    dot_color=dot_color
                )
    
    # Compute differences
    diffs = {
        "FST12_ZeroShot": np.mean(results[("FST12", "ZeroShot", "Dots")]) - np.mean(results[("FST12", "ZeroShot", "NoDots")]),
        "FST56_ZeroShot": np.mean(results[("FST56", "ZeroShot", "Dots")]) - np.mean(results[("FST56", "ZeroShot", "NoDots")]),
        "FST12_ICL": np.mean(results[("FST12", "ICL", "Dots")]) - np.mean(results[("FST12", "ICL", "NoDots")]),
        "FST56_ICL": np.mean(results[("FST56", "ICL", "Dots")]) - np.mean(results[("FST56", "ICL", "NoDots")]),
    }
    
    # Standard errors via bootstrap
    def bootstrap_diff_se(a, b, n_boot=1000):
        diffs_list = []
        for _ in range(n_boot):
            a_sample = np.random.choice(a, size=len(a), replace=True)
            b_sample = np.random.choice(b, size=len(b), replace=True)
            diffs_list.append(np.mean(a_sample) - np.mean(b_sample))
        return np.std(diffs_list)
    
    ses = {
        "FST12_ZeroShot": bootstrap_diff_se(results[("FST12", "ZeroShot", "Dots")], results[("FST12", "ZeroShot", "NoDots")]),
        "FST56_ZeroShot": bootstrap_diff_se(results[("FST56", "ZeroShot", "Dots")], results[("FST56", "ZeroShot", "NoDots")]),
        "FST12_ICL": bootstrap_diff_se(results[("FST12", "ICL", "Dots")], results[("FST12", "ICL", "NoDots")]),
        "FST56_ICL": bootstrap_diff_se(results[("FST56", "ICL", "Dots")], results[("FST56", "ICL", "NoDots")]),
    }
    
    # Print results
    print("\n" + "="*50)
    print("Results: Δ logprob(Malignant) after adding purple dots")
    print("="*50)
    for k, v in diffs.items():
        print(f"{k}: {v:.4f} ± {ses[k]:.4f}")
    
    # Create the new combined plot with examples and bar charts
    plot_results_with_examples(
        fst12_images, fst56_images,
        diffs, ses,
        args.output,
        seed=args.seed
    )


if __name__ == "__main__":
    main()