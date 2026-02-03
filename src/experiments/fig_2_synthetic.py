#!/usr/bin/env python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from interpretability.vcr import ConceptAnalyzer, PromptTemplate
from models.flamingo import FlamingoAPI
from interpretability.utils import CLIPEmbedder, compute_inner_products, LayerOverride


# =============================================================================
# STYLING
# =============================================================================

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


# =============================================================================
# FEATURE PAIR DEFINITIONS
# =============================================================================

FEATURE_PAIRS = {
    'red_green': {
        'feature_a': 'red',
        'feature_b': 'green',
        'concept_a': 'red',
        'concept_b': 'green',
        'color_a': NORD['red'],
        'color_b': NORD['green'],
    },
    'left_right': {
        'feature_a': 'left',
        'feature_b': 'right',
        'concept_a': 'left',
        'concept_b': 'right',
        'color_a': NORD['purple'],
        'color_b': NORD['orange'],
    },
    'one_many': {
        'feature_a': 'one',
        'feature_b': 'many',
        'concept_a': 'one',
        'concept_b': 'many',
        'color_a': NORD['frost'],
        'color_b': NORD['red'],
    },
    'horizontal_vertical': {
        'feature_a': 'horizontal',
        'feature_b': 'vertical',
        'concept_a': 'horizontal',
        'concept_b': 'vertical',
        'color_a': NORD['green'],
        'color_b': NORD['purple'],
    },
    'square_circle': {
        'feature_a': 'square',
        'feature_b': 'circle',
        'concept_a': 'square',
        'concept_b': 'circle',
        'color_a': NORD['blue'],
        'color_b': NORD['sand'],
    },
    'empty_filled': {
        'feature_a': 'empty',
        'feature_b': 'filled',
        'concept_a': 'loops',      # VCR concept name for empty
        'concept_b': 'spots',      # VCR concept name for filled
        'color_a': NORD['ice'],
        'color_b': NORD['blue'],
    },
    'striped_solid': {
        'feature_a': 'striped',
        'feature_b': 'solid',
        'concept_a': 'bars',       # VCR concept name for striped
        'concept_b': 'cube',       # VCR concept name for solid
        'color_a': NORD['ice'],
        'color_b': NORD['dark'],
    },
    'top_bottom': {
        'feature_a': 'top',
        'feature_b': 'bottom',
        'concept_a': 'top',
        'concept_b': 'bottom',
        'color_a': NORD['frost'],
        'color_b': NORD['purple'],
    },
}


# =============================================================================
# IMAGE GENERATION FOR EACH FEATURE PAIR
# =============================================================================

def generate_image_red_green(has_a, has_b, size=224, seed=None):
    """Red object vs Green object."""
    if seed is not None:
        np.random.seed(seed)
    
    img = Image.new('RGB', (size, size), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    
    if has_a:  # red
        x, y = np.random.randint(10, size//2-30, 2)
        draw.rectangle([x, y, x+60, y+60], fill=(220, 50, 50), outline='darkred', width=2)
    
    if has_b:  # green
        x, y = np.random.randint(size//2, size-70, 2)
        draw.rectangle([x, y, x+60, y+60], fill=(50, 180, 50), outline='darkgreen', width=2)
    
    for _ in range(2):
        x, y = np.random.randint(10, size-40, 2)
        draw.ellipse([x, y, x+25, y+25], fill=(150, 150, 180))
    
    return img


def generate_image_left_right(has_a, has_b, size=224, seed=None):
    """Object on left vs Object on right."""
    if seed is not None:
        np.random.seed(seed)
    
    img = Image.new('RGB', (size, size), color=(230, 230, 220))
    draw = ImageDraw.Draw(img)
    
    color = tuple(np.random.randint(50, 150, 3).tolist())
    obj_size = 50
    
    if has_a:  # left
        x = np.random.randint(10, size//2 - obj_size - 10)
        y = np.random.randint(30, size - obj_size - 30)
        draw.ellipse([x, y, x+obj_size, y+obj_size], fill=color, outline='black', width=2)
    
    if has_b:  # right
        x = np.random.randint(size//2 + 10, size - obj_size - 10)
        y = np.random.randint(30, size - obj_size - 30)
        draw.ellipse([x, y, x+obj_size, y+obj_size], fill=color, outline='black', width=2)
    
    return img


def generate_image_one_many(has_a, has_b, size=224, seed=None):
    """One object vs Many objects."""
    if seed is not None:
        np.random.seed(seed)
    
    img = Image.new('RGB', (size, size), color=(225, 225, 235))
    draw = ImageDraw.Draw(img)
    
    color = tuple(np.random.randint(50, 150, 3).tolist())
    
    if has_a and not has_b:  # one only
        x, y = np.random.randint(60, size-100, 2)
        draw.ellipse([x, y, x+50, y+50], fill=color, outline='black', width=2)
    elif has_b and not has_a:  # many only
        for _ in range(np.random.randint(6, 10)):
            x, y = np.random.randint(10, size-30, 2)
            draw.ellipse([x, y, x+20, y+20], fill=color, outline='black', width=1)
    elif has_a and has_b:  # both
        x, y = np.random.randint(10, size//2-50, 2)
        draw.ellipse([x, y, x+50, y+50], fill=color, outline='black', width=2)
        for _ in range(np.random.randint(5, 8)):
            x, y = np.random.randint(size//2, size-25, 2)
            draw.ellipse([x, y, x+18, y+18], fill=color, outline='black', width=1)
    
    return img


def generate_image_horizontal_vertical(has_a, has_b, size=224, seed=None):
    """Horizontal line/bar vs Vertical line/bar."""
    if seed is not None:
        np.random.seed(seed)
    
    img = Image.new('RGB', (size, size), color=(235, 230, 225))
    draw = ImageDraw.Draw(img)
    
    color = tuple(np.random.randint(30, 100, 3).tolist())
    
    if has_a:  # horizontal
        y = np.random.randint(30, size//2 - 20)
        x1 = np.random.randint(20, 60)
        x2 = np.random.randint(size-80, size-20)
        draw.rectangle([x1, y, x2, y+15], fill=color, outline='black', width=1)
    
    if has_b:  # vertical
        x = np.random.randint(size//2, size - 40)
        y1 = np.random.randint(20, 60)
        y2 = np.random.randint(size-80, size-20)
        draw.rectangle([x, y1, x+15, y2], fill=color, outline='black', width=1)
    
    return img


def generate_image_square_circle(has_a, has_b, size=224, seed=None):
    """Square shape vs Circle shape."""
    if seed is not None:
        np.random.seed(seed)
    
    img = Image.new('RGB', (size, size), color=(220, 220, 220))
    draw = ImageDraw.Draw(img)
    
    color = tuple(np.random.randint(50, 150, 3).tolist())
    obj_size = 60
    
    if has_a:  # square
        x = np.random.randint(20, size//2 - obj_size - 10)
        y = np.random.randint(40, size - obj_size - 40)
        draw.rectangle([x, y, x+obj_size, y+obj_size], fill=color, outline='black', width=3)
    
    if has_b:  # circle
        x = np.random.randint(size//2 + 10, size - obj_size - 20)
        y = np.random.randint(40, size - obj_size - 40)
        draw.ellipse([x, y, x+obj_size, y+obj_size], fill=color, outline='black', width=3)
    
    return img


def generate_image_empty_filled(has_a, has_b, size=224, seed=None):
    """Empty (hollow/outline) shape vs Filled (solid) shape."""
    if seed is not None:
        np.random.seed(seed)
    
    img = Image.new('RGB', (size, size), color=(220, 220, 220))
    draw = ImageDraw.Draw(img)
    
    color = tuple(np.random.randint(40, 120, 3).tolist())
    obj_size = 60
    
    if has_a:  # empty/hollow
        x = np.random.randint(20, size//2 - obj_size - 10)
        y = np.random.randint(40, size - obj_size - 40)
        # Draw outline only (no fill)
        draw.ellipse([x, y, x+obj_size, y+obj_size], fill=None, outline=color, width=5)
    
    if has_b:  # filled/solid
        x = np.random.randint(size//2 + 10, size - obj_size - 20)
        y = np.random.randint(40, size - obj_size - 40)
        # Draw filled shape
        draw.ellipse([x, y, x+obj_size, y+obj_size], fill=color, outline=color, width=2)
    
    return img


def generate_image_striped_solid(has_a, has_b, size=224, seed=None):
    """Striped pattern vs Solid fill."""
    if seed is not None:
        np.random.seed(seed)
    
    img = Image.new('RGB', (size, size), color=(215, 215, 220))
    draw = ImageDraw.Draw(img)
    
    base_color = tuple(np.random.randint(60, 140, 3).tolist())
    obj_size = 70
    
    if has_a:  # striped
        x = np.random.randint(15, size//2 - obj_size - 5)
        y = np.random.randint(30, size - obj_size - 30)
        # Draw rectangle with stripes
        draw.rectangle([x, y, x+obj_size, y+obj_size], fill=(240, 240, 240), outline='black', width=2)
        stripe_color = base_color
        for i in range(0, obj_size, 8):
            draw.line([(x+i, y), (x+i, y+obj_size)], fill=stripe_color, width=3)
    
    if has_b:  # solid
        x = np.random.randint(size//2 + 5, size - obj_size - 15)
        y = np.random.randint(30, size - obj_size - 30)
        draw.rectangle([x, y, x+obj_size, y+obj_size], fill=base_color, outline='black', width=2)
    
    return img


def generate_image_top_bottom(has_a, has_b, size=224, seed=None):
    """Object at top vs Object at bottom."""
    if seed is not None:
        np.random.seed(seed)
    
    img = Image.new('RGB', (size, size), color=(225, 225, 215))
    draw = ImageDraw.Draw(img)
    
    color = tuple(np.random.randint(50, 150, 3).tolist())
    obj_size = 50
    
    if has_a:  # top
        x = np.random.randint(40, size - obj_size - 40)
        y = np.random.randint(15, size//2 - obj_size - 15)
        draw.ellipse([x, y, x+obj_size, y+obj_size], fill=color, outline='black', width=2)
    
    if has_b:  # bottom
        x = np.random.randint(40, size - obj_size - 40)
        y = np.random.randint(size//2 + 15, size - obj_size - 15)
        draw.ellipse([x, y, x+obj_size, y+obj_size], fill=color, outline='black', width=2)
    
    return img


GENERATORS = {
    'red_green': generate_image_red_green,
    'left_right': generate_image_left_right,
    'one_many': generate_image_one_many,
    'horizontal_vertical': generate_image_horizontal_vertical,
    'square_circle': generate_image_square_circle,
    'empty_filled': generate_image_empty_filled,
    'striped_solid': generate_image_striped_solid,
    'top_bottom': generate_image_top_bottom,
}


# =============================================================================
# DATASET CREATION
# =============================================================================

def create_dataset(output_dir, feature_pair, corr_a, corr_b, n_train=400):
    """Create dataset with specified correlations for the feature pair."""
    output_dir = Path(output_dir)
    (output_dir / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'test').mkdir(parents=True, exist_ok=True)
    
    generator = GENERATORS[feature_pair]
    
    n_pos = n_train // 2
    n_neg = n_train // 2
    
    n_a_in_pos = int(n_pos * (corr_a + 1) / 2)
    n_b_in_pos = int(n_pos * (corr_b + 1) / 2)
    n_a_in_neg = int(n_neg * (1 - (corr_a + 1) / 2))
    n_b_in_neg = int(n_neg * (1 - (corr_b + 1) / 2))
    
    train_data = []
    idx = 0
    
    for i in range(n_pos):
        has_a = i < n_a_in_pos
        has_b = i < n_b_in_pos
        img = generator(has_a, has_b, seed=idx)
        filename = f'train_{idx:04d}.png'
        img.save(output_dir / 'train' / filename)
        train_data.append({
            'filename': filename,
            'has_a': has_a,
            'has_b': has_b,
            'label': 'positive'
        })
        idx += 1
    
    for i in range(n_neg):
        has_a = i < n_a_in_neg
        has_b = i < n_b_in_neg
        img = generator(has_a, has_b, seed=idx)
        filename = f'train_{idx:04d}.png'
        img.save(output_dir / 'train' / filename)
        train_data.append({
            'filename': filename,
            'has_a': has_a,
            'has_b': has_b,
            'label': 'negative'
        })
        idx += 1
    
    # Balanced test set
    test_data = []
    idx = 0
    for has_a in [True, False]:
        for has_b in [True, False]:
            for _ in range(50):
                img = generator(has_a, has_b, seed=idx + 10000)
                filename = f'test_{idx:04d}.png'
                img.save(output_dir / 'test' / filename)
                test_data.append({
                    'filename': filename,
                    'has_a': has_a,
                    'has_b': has_b,
                    'label': 'positive'
                })
                idx += 1
    
    pd.DataFrame(train_data).to_csv(output_dir / 'train_metadata.csv', index=False)
    pd.DataFrame(test_data).to_csv(output_dir / 'test_metadata.csv', index=False)
    
    return train_data, test_data


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, metadata, base_dir, image_processor):
        self.metadata = metadata
        self.base_dir = Path(base_dir)
        self.image_processor = image_processor
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img = Image.open(self.base_dir / row['filename']).convert('RGB')
        return {'image': self.image_processor(img), 'label': row['label']}


# =============================================================================
# MODEL TRAINING AND EVALUATION
# =============================================================================

def finetune(model, train_dataset, n_epochs=5, finetune_layer_offset=-1):
    """Fine-tune specified layer.
    
    Args:
        model: FlamingoAPI model
        train_dataset: Training dataset
        n_epochs: Number of training epochs
        finetune_layer_offset: Which layer to unfreeze (negative index from end)
                              -1 = last layer, -4 = fourth from last, etc.
    """
    # Freeze all parameters
    for param in model.model.parameters():
        param.requires_grad = False
    
    # Unfreeze specified layer using negative indexing
    if model.model_name == 'OpenFlamingo-3B-Instruct':
        blocks = model.model.lang_encoder.transformer.blocks
    else:
        blocks = model.model.lang_encoder.gpt_neox.layers
        
    layer_idx = len(blocks) + finetune_layer_offset  # Convert negative to positive index
    print(f"  Fine-tuning layer {layer_idx} (offset {finetune_layer_offset} from end, total {len(blocks)} blocks)")
    
    for param in blocks[layer_idx].parameters():
        param.requires_grad = True
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    optimizer = torch.optim.AdamW([p for p in model.model.parameters() if p.requires_grad], lr=1e-4)
    loss_fct = nn.CrossEntropyLoss()
    
    for epoch in range(n_epochs):
        model.model.train()
        for batch in train_loader:
            images = batch['image'].cuda().unsqueeze(1).unsqueeze(2)
            prompts = [f"<image>This image is {label}" for label in batch['label']]
            encoded = model.tokenizer(prompts, padding=True, return_tensors='pt')
            outputs = model.model(vision_x=images, lang_x=encoded['input_ids'].cuda(),
                                attention_mask=encoded['attention_mask'].cuda())
            loss = loss_fct(outputs.logits[..., :-1, :].contiguous().view(-1, outputs.logits.size(-1)),
                          encoded['input_ids'][..., 1:].contiguous().view(-1).cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del images, outputs, loss
    
    del optimizer, train_loader
    return model


def compute_interventional_effects(model, test_dataset, test_df):
    """Compute interventional effects for features A and B."""
    model.model.eval()
    
    pos_token_id = model.tokenizer.encode(" positive", add_special_tokens=False)[0]
    neg_token_id = model.tokenizer.encode(" negative", add_special_tokens=False)[0]
    
    results = []
    
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            item = test_dataset[idx]
            row = test_df.iloc[idx]
            
            image = item['image'].cuda().unsqueeze(0).unsqueeze(1).unsqueeze(2)
            prompt = "<image>This image is"
            encoded = model.tokenizer(prompt, return_tensors='pt')
            
            outputs = model.model(
                vision_x=image,
                lang_x=encoded['input_ids'].cuda(),
                attention_mask=encoded['attention_mask'].cuda()
            )
            
            next_token_logits = outputs.logits[0, -1, :]
            probs = torch.softmax(next_token_logits, dim=0)
            
            pos_prob = probs[pos_token_id].item()
            neg_prob = probs[neg_token_id].item()
            pos_prob_normalized = pos_prob / (pos_prob + neg_prob)
            
            results.append({
                'has_a': row['has_a'],
                'has_b': row['has_b'],
                'pos_prob_normalized': pos_prob_normalized
            })
            
            del image, outputs
    
    results_df = pd.DataFrame(results)
    
    a_present = results_df[results_df['has_a'] == True]['pos_prob_normalized'].mean()
    a_absent = results_df[results_df['has_a'] == False]['pos_prob_normalized'].mean()
    a_effect = a_present - a_absent
    
    b_present = results_df[results_df['has_b'] == True]['pos_prob_normalized'].mean()
    b_absent = results_df[results_df['has_b'] == False]['pos_prob_normalized'].mean()
    b_effect = b_present - b_absent
    
    return {'a_effect': a_effect, 'b_effect': b_effect}, results_df.pos_prob_normalized.values


def compute_clip_scores(sim_matrix, per_image_probs):
    """
    Compute CLIP score for each concept as correlation with model output.
    
    For each concept c:
        clip_score[c] = pearsonr(sim_matrix[c, :], per_image_probs)
    
    Args:
        sim_matrix: [n_concepts, n_images] tensor of CLIP similarities
        per_image_probs: [n_images] array of model P(positive) for each image
    
    Returns:
        np.array of shape [n_concepts] with correlation for each concept
    """
    sim_np = sim_matrix.cpu().numpy()  # [n_concepts, n_images]
    n_concepts = sim_np.shape[0]
    
    clip_scores = np.zeros(n_concepts)
    
    for c in range(n_concepts):
        r, _ = stats.pearsonr(sim_np[c, :], per_image_probs)
        clip_scores[c] = r
    
    return clip_scores


def run_vcr(model, model_name, layer_name, data_dir, concept_files, clip):
    """Run VCR analysis."""
    test_df = pd.read_csv(Path(data_dir) / 'test_metadata.csv')
    test_paths = [str(Path(data_dir) / 'test' / f) for f in test_df['filename']]
    
    analyzer = ConceptAnalyzer.__new__(ConceptAnalyzer)
    analyzer.model = model
    analyzer.model_name = model_name
    analyzer.clip = clip
    analyzer.image_processor = model.image_processor
    analyzer.wrapped_layer = None
    analyzer.concept_model = None
    analyzer.concept_vectors = None
    analyzer.model.model.train()
    analyzer.setup_layer_hook(layer_name, LayerOverride)
    
    test_dataset = SyntheticDataset(test_df, Path(data_dir) / 'test', model.image_processor)
    image_emb, text_emb, concept_texts = analyzer.get_embeddings(test_paths, concept_files)
    sim_matrix = compute_inner_products(text_emb, image_emb)
    
    prompt_template = PromptTemplate(base_prompt="", query_template="<image>This image is")
    activations = analyzer.collect_activations(test_dataset, prompt_template, batch_size=1)
    analyzer.train_concept_model(activations, sim_matrix)
    concept_vectors = analyzer.extract_concept_vectors()
    concept_weights = analyzer.compute_concept_weights(sim_matrix)
    
    original_calc = analyzer.calculate_directional_derivatives
    def patched_calc(dataset, concept_vectors, concept_weights, prompt_template, completion,
                    demo_paths=None, demo_labels=None):
        param_states = {name: param.requires_grad for name, param in analyzer.model.model.named_parameters()}
        for param in analyzer.model.model.parameters():
            param.requires_grad = True
        try:
            result = original_calc(dataset, concept_vectors, concept_weights, prompt_template,
                                 completion, demo_paths, demo_labels)
        finally:
            for name, param in analyzer.model.model.named_parameters():
                param.requires_grad = param_states[name]
        return result
    
    analyzer.calculate_directional_derivatives = patched_calc
    
    with torch.enable_grad():
        _, raw_sens = analyzer.calculate_directional_derivatives(
            test_dataset, concept_vectors, concept_weights, prompt_template, " positive"
        )
    
    return concept_texts, raw_sens, sim_matrix


def find_concept_idx(concept_texts, target):
    """Find index of target concept."""
    concept_lower = [c.lower().strip() for c in concept_texts]
    try:
        return concept_lower.index(target.lower())
    except ValueError:
        for i, c in enumerate(concept_lower):
            if target.lower() in c:
                return i
        return None


# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================

def run_experiment(feature_pair, corr_values, n_seeds=10, model_name='OpenFlamingo-3B-Instruct',
                   finetune_layer_offset=-1):
    """Run experiment for one feature pair across all correlation conditions."""
    
    pair_info = FEATURE_PAIRS[feature_pair]
    data_dir = f'./synthetic_data_{feature_pair}'
    if model_name=='OpenFlamingo-3B-Instruct':
        layer_name = 'model.lang_encoder.transformer.blocks.23.decoder_layer'  # Always use last layer for interpretability
    else:
        layer_name = 'model.lang_encoder.gpt_neox.layers.31.decoder_layer' # Always use last layer for interpretability
    concept_files = ['/home/joseph/vlm-interp/src/concept_sets/google-10000-english-no-swears.txt']
    
    print(f"\n{'='*70}")
    print(f"FEATURE PAIR: {feature_pair}")
    print(f"  Feature A: {pair_info['feature_a']} (VCR concept: {pair_info['concept_a']})")
    print(f"  Feature B: {pair_info['feature_b']} (VCR concept: {pair_info['concept_b']})")
    print(f"  Fine-tuning layer offset: {finetune_layer_offset}")
    print(f"{'='*70}")
    
    clip = CLIPEmbedder()
    
    all_results = []
    concept_texts = None
    
    for corr_a in corr_values:
        corr_b = -corr_a
        
        print(f"\n  Correlation: A={corr_a:+.1f}, B={corr_b:+.1f}")
        create_dataset(data_dir, feature_pair, corr_a, corr_b)
        
        train_df = pd.read_csv(Path(data_dir) / 'train_metadata.csv')
        test_df = pd.read_csv(Path(data_dir) / 'test_metadata.csv')
        test_paths = [str(Path(data_dir) / 'test' / f) for f in test_df['filename']]
        
        for seed in tqdm(range(n_seeds), desc=f"    Seeds", leave=False):
            model = FlamingoAPI(model_name)
            
            train_resampled = train_df.sample(n=len(train_df), replace=True, random_state=seed)
            train_dataset = SyntheticDataset(train_resampled, Path(data_dir) / 'train', model.image_processor)
            test_dataset = SyntheticDataset(test_df, Path(data_dir) / 'test', model.image_processor)
            
            model = finetune(model, train_dataset, finetune_layer_offset=finetune_layer_offset)
            
            effects, per_image_probs = compute_interventional_effects(model, test_dataset, test_df)
            
            seed_concepts, raw_sens, sim_matrix = run_vcr(model, model_name, layer_name, data_dir, concept_files, clip)
            
            # CLIP-only baseline: mean similarity across test images (from sim_matrix)
            clip_scores = compute_clip_scores(sim_matrix, per_image_probs)
            
            if concept_texts is None:
                concept_texts = seed_concepts
            
            idx_a = find_concept_idx(concept_texts, pair_info['concept_a'])
            idx_b = find_concept_idx(concept_texts, pair_info['concept_b'])
            
            if idx_a is not None:
                all_results.append({
                    'feature_pair': feature_pair,
                    'corr_a': corr_a,
                    'feature': pair_info['feature_a'],
                    'vcr_sensitivity': raw_sens.mean(axis=0)[idx_a],
                    'clip_score': clip_scores[idx_a],
                    'interventional_effect': effects['a_effect'],
                    'seed': seed,
                    'finetune_layer': finetune_layer_offset
                })
            
            if idx_b is not None:
                all_results.append({
                    'feature_pair': feature_pair,
                    'corr_a': corr_a,
                    'feature': pair_info['feature_b'],
                    'vcr_sensitivity': raw_sens.mean(axis=0)[idx_b],
                    'clip_score': clip_scores[idx_b],
                    'interventional_effect': effects['b_effect'],
                    'seed': seed,
                    'finetune_layer': finetune_layer_offset
                })
            
            # Cleanup
            del model.model, model.tokenizer, model.image_processor, model
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    
    return pd.DataFrame(all_results)


# =============================================================================
# PLOTTING
# =============================================================================

def plot_single_layer_results(all_df, output_dir, layer_label):
    """Generate plots for a single layer's results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Plot 1: Combined scatter for all feature pairs - VCR
    fig, ax = plt.subplots(figsize=(10, 8))
    
    feature_pairs = all_df['feature_pair'].unique()
    
    for fp in feature_pairs:
        fp_data = all_df[all_df['feature_pair'] == fp]
        pair_info = FEATURE_PAIRS[fp]
        
        for feat in [pair_info['feature_a'], pair_info['feature_b']]:
            feat_data = fp_data[fp_data['feature'] == feat]
            color = pair_info['color_a'] if feat == pair_info['feature_a'] else pair_info['color_b']
            ax.scatter(feat_data['vcr_sensitivity'], feat_data['interventional_effect'],
                      alpha=0.5, s=30, c=color, label=f'{fp}: {feat}', edgecolor='white', linewidth=0.3)
    
    slope, intercept, r_value, p_value, _ = stats.linregress(
        all_df['vcr_sensitivity'], all_df['interventional_effect']
    )
    x_line = np.linspace(all_df['vcr_sensitivity'].min(), all_df['vcr_sensitivity'].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=2, label=f'r={r_value:.3f}')
    
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('VCR Sensitivity (predicted)')
    ax.set_ylabel('Interventional Effect (actual)')
    ax.set_title(f'VCR vs Interventional Effects (Fine-tune Layer {layer_label})\n$R^2$={r_value**2:.3f}, p={p_value:.2e}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'all_features_scatter_layer{layer_label}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / f'all_features_scatter_layer{layer_label}.png'}")
    plt.close()
    
    # Plot 2: Per-feature-pair subplots - VCR vs CLIP comparison
    n_pairs = len(feature_pairs)
    n_cols = 3
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.flatten()
    
    for i, fp in enumerate(feature_pairs):
        ax = axes[i]
        fp_data = all_df[all_df['feature_pair'] == fp]
        pair_info = FEATURE_PAIRS[fp]
        
        for feat in [pair_info['feature_a'], pair_info['feature_b']]:
            feat_data = fp_data[fp_data['feature'] == feat]
            color = pair_info['color_a'] if feat == pair_info['feature_a'] else pair_info['color_b']
            ax.scatter(feat_data['vcr_sensitivity'], feat_data['interventional_effect'],
                      alpha=0.6, s=40, c=color, label=feat, edgecolor='white', linewidth=0.5)
        
        if len(fp_data) > 2:
            slope, _, r, p, _ = stats.linregress(fp_data['vcr_sensitivity'], fp_data['interventional_effect'])
            ax.set_title(f'{fp}\nr={r:.3f}')
        else:
            ax.set_title(fp)
        
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
        ax.legend(fontsize=8)
        ax.set_xlabel('VCR')
        ax.set_ylabel('Intervention')
    
    for i in range(len(feature_pairs), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Fine-tune Layer {layer_label}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'per_feature_pair_scatter_layer{layer_label}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / f'per_feature_pair_scatter_layer{layer_label}.png'}")
    plt.close()
    
    return r_value, p_value


def plot_vcr_vs_clip_comparison(all_df, output_dir):
    """Generate comparison plots between VCR and CLIP-only baseline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    feature_pairs = all_df['feature_pair'].unique()
    
    # Plot 1: Side-by-side scatter - VCR vs CLIP
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # VCR scatter
    ax = axes[0]
    for fp in feature_pairs:
        fp_data = all_df[all_df['feature_pair'] == fp]
        pair_info = FEATURE_PAIRS[fp]
        for feat in [pair_info['feature_a'], pair_info['feature_b']]:
            feat_data = fp_data[fp_data['feature'] == feat]
            color = pair_info['color_a'] if feat == pair_info['feature_a'] else pair_info['color_b']
            ax.scatter(feat_data['vcr_sensitivity'], feat_data['interventional_effect'],
                      alpha=0.5, s=30, c=color, edgecolor='white', linewidth=0.3)
    
    slope, intercept, r_vcr, p_vcr, _ = stats.linregress(
        all_df['vcr_sensitivity'], all_df['interventional_effect']
    )
    x_line = np.linspace(all_df['vcr_sensitivity'].min(), all_df['vcr_sensitivity'].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=2)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('VCR Sensitivity')
    ax.set_ylabel('Interventional Effect')
    ax.set_title(f'VCR (gradient-based)\nr={r_vcr:.3f}, $R^2$={r_vcr**2:.3f}')
    
    # CLIP scatter
    ax = axes[1]
    for fp in feature_pairs:
        fp_data = all_df[all_df['feature_pair'] == fp]
        pair_info = FEATURE_PAIRS[fp]
        for feat in [pair_info['feature_a'], pair_info['feature_b']]:
            feat_data = fp_data[fp_data['feature'] == feat]
            color = pair_info['color_a'] if feat == pair_info['feature_a'] else pair_info['color_b']
            ax.scatter(feat_data['clip_score'], feat_data['interventional_effect'],
                      alpha=0.5, s=30, c=color, edgecolor='white', linewidth=0.3)
    
    slope, intercept, r_clip, p_clip, _ = stats.linregress(
        all_df['clip_score'], all_df['interventional_effect']
    )
    x_line = np.linspace(all_df['clip_score'].min(), all_df['clip_score'].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=2)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('CLIP Correlation')
    ax.set_ylabel('Interventional Effect')
    ax.set_title(f'CLIP-only (no gradients)\nr={r_clip:.3f}, $R^2$={r_clip**2:.3f}')
    
    plt.suptitle('VCR vs CLIP-only Baseline', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'vcr_vs_clip_scatter.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'vcr_vs_clip_scatter.png'}")
    plt.close()
    
    # Plot 2: Per-feature-pair correlation comparison (bar chart)
    correlations = []
    for fp in feature_pairs:
        fp_data = all_df[all_df['feature_pair'] == fp]
        if len(fp_data) > 2:
            r_vcr_fp, p_vcr_fp = stats.pearsonr(fp_data['vcr_sensitivity'], fp_data['interventional_effect'])
            r_clip_fp, p_clip_fp = stats.pearsonr(fp_data['clip_score'], fp_data['interventional_effect'])
            correlations.append({
                'feature_pair': fp,
                'method': 'VCR',
                'r': r_vcr_fp,
                'p': p_vcr_fp
            })
            correlations.append({
                'feature_pair': fp,
                'method': 'CLIP',
                'r': r_clip_fp,
                'p': p_clip_fp
            })
    
    corr_df = pd.DataFrame(correlations)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(feature_pairs))
    width = 0.35
    
    vcr_r = corr_df[corr_df['method'] == 'VCR']['r'].values
    clip_r = corr_df[corr_df['method'] == 'CLIP']['r'].values
    
    bars1 = ax.bar(x - width/2, vcr_r, width, label='VCR (gradient-based)', color=NORD['blue'], alpha=0.8)
    bars2 = ax.bar(x + width/2, clip_r, width, label='CLIP-only', color=NORD['orange'], alpha=0.8)
    
    ax.set_ylabel('Pearson r')
    ax.set_title('Correlation with Interventional Effects: VCR vs CLIP-only')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_pairs, rotation=45, ha='right')
    ax.legend()
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_ylim(-1, 1)
    
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
    plt.savefig(output_dir / 'vcr_vs_clip_by_feature.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'vcr_vs_clip_by_feature.png'}")
    plt.close()
    
    # Plot 3: Overall summary bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    
    r_vcr_all, p_vcr_all = stats.pearsonr(all_df['vcr_sensitivity'], all_df['interventional_effect'])
    r_clip_all, p_clip_all = stats.pearsonr(all_df['clip_score'], all_df['interventional_effect'])
    
    methods = ['VCR\n(gradient-based)', 'CLIP-only']
    r_values = [r_vcr_all, r_clip_all]
    r2_values = [r_vcr_all**2, r_clip_all**2]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, r_values, width, label='Pearson r', color=NORD['blue'], alpha=0.8)
    bars2 = ax.bar(x + width/2, r2_values, width, label='$R^2$', color=NORD['green'], alpha=0.8)
    
    ax.set_ylabel('Correlation')
    ax.set_title('Overall Correlation with Interventional Effects')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 1)
    
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
    plt.savefig(output_dir / 'vcr_vs_clip_overall.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'vcr_vs_clip_overall.png'}")
    plt.close()
    
    return corr_df, r_vcr_all, r_clip_all


def plot_layer_comparison(df_layer1, df_layer4, output_dir):
    """Generate comparison plots between layer -1 and layer -4 results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Add layer labels
    df_layer1 = df_layer1.copy()
    df_layer4 = df_layer4.copy()
    df_layer1['layer'] = 'Layer -1 (last)'
    df_layer4['layer'] = 'Layer -4'
    
    combined = pd.concat([df_layer1, df_layer4], ignore_index=True)
    
    # Plot 1: Side-by-side scatter comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, (layer_name, layer_df) in zip(axes, [('Layer -1 (last)', df_layer1), ('Layer -4', df_layer4)]):
        feature_pairs = layer_df['feature_pair'].unique()
        
        for fp in feature_pairs:
            fp_data = layer_df[layer_df['feature_pair'] == fp]
            pair_info = FEATURE_PAIRS[fp]
            
            for feat in [pair_info['feature_a'], pair_info['feature_b']]:
                feat_data = fp_data[fp_data['feature'] == feat]
                color = pair_info['color_a'] if feat == pair_info['feature_a'] else pair_info['color_b']
                ax.scatter(feat_data['vcr_sensitivity'], feat_data['interventional_effect'],
                          alpha=0.5, s=30, c=color, edgecolor='white', linewidth=0.3)
        
        slope, intercept, r_value, p_value, _ = stats.linregress(
            layer_df['vcr_sensitivity'], layer_df['interventional_effect']
        )
        x_line = np.linspace(layer_df['vcr_sensitivity'].min(), layer_df['vcr_sensitivity'].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=2)
        
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('VCR Sensitivity')
        ax.set_ylabel('Interventional Effect')
        ax.set_title(f'{layer_name}\nr={r_value:.3f}, $R^2$={r_value**2:.3f}')
    
    plt.suptitle('VCR Validation: Fine-tuning Layer Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'layer_comparison_scatter.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'layer_comparison_scatter.png'}")
    plt.close()
    
    # Plot 2: Per-feature-pair correlation comparison
    feature_pairs = df_layer1['feature_pair'].unique()
    
    correlations = []
    for fp in feature_pairs:
        for layer_name, layer_df in [('-1', df_layer1), ('-4', df_layer4)]:
            fp_data = layer_df[layer_df['feature_pair'] == fp]
            if len(fp_data) > 2:
                r, p = stats.pearsonr(fp_data['vcr_sensitivity'], fp_data['interventional_effect'])
                correlations.append({'feature_pair': fp, 'layer': layer_name, 'r': r, 'p': p})
    
    corr_df = pd.DataFrame(correlations)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(feature_pairs))
    width = 0.35
    
    layer1_r = corr_df[corr_df['layer'] == '-1']['r'].values
    layer4_r = corr_df[corr_df['layer'] == '-4']['r'].values
    
    bars1 = ax.bar(x - width/2, layer1_r, width, label='Layer -1 (last)', color=NORD['blue'], alpha=0.8)
    bars2 = ax.bar(x + width/2, layer4_r, width, label='Layer -4', color=NORD['orange'], alpha=0.8)
    
    ax.set_ylabel('Pearson r')
    ax.set_title('VCR-Intervention Correlation by Feature Pair and Fine-tuning Layer')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_pairs, rotation=45, ha='right')
    ax.legend()
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_ylim(-1, 1)
    
    # Add significance markers
    for i, (r1, r4) in enumerate(zip(layer1_r, layer4_r)):
        p1 = corr_df[(corr_df['layer'] == '-1') & (corr_df['feature_pair'] == feature_pairs[i])]['p'].values[0]
        p4 = corr_df[(corr_df['layer'] == '-4') & (corr_df['feature_pair'] == feature_pairs[i])]['p'].values[0]
        
        sig1 = '***' if p1 < 0.001 else '**' if p1 < 0.01 else '*' if p1 < 0.05 else ''
        sig4 = '***' if p4 < 0.001 else '**' if p4 < 0.01 else '*' if p4 < 0.05 else ''
        
        ax.text(i - width/2, r1 + 0.03, sig1, ha='center', fontsize=8)
        ax.text(i + width/2, r4 + 0.03, sig4, ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'layer_comparison_by_feature.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'layer_comparison_by_feature.png'}")
    plt.close()
    
    # Plot 3: Overall comparison summary
    fig, ax = plt.subplots(figsize=(8, 6))
    
    r1, p1 = stats.pearsonr(df_layer1['vcr_sensitivity'], df_layer1['interventional_effect'])
    r4, p4 = stats.pearsonr(df_layer4['vcr_sensitivity'], df_layer4['interventional_effect'])
    
    layers = ['Layer -1\n(last)', 'Layer -4']
    r_values = [r1, r4]
    r2_values = [r1**2, r4**2]
    
    x = np.arange(len(layers))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, r_values, width, label='Pearson r', color=NORD['blue'], alpha=0.8)
    bars2 = ax.bar(x + width/2, r2_values, width, label='$R^2$', color=NORD['green'], alpha=0.8)
    
    ax.set_ylabel('Correlation')
    ax.set_title('Overall VCR-Intervention Correlation by Fine-tuning Layer')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bar, val in zip(bars1, r_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', 
                ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, r2_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', 
                ha='center', va='bottom', fontsize=10)
    
    # Add p-value annotations
    ax.text(0, -0.1, f'p={p1:.2e}', ha='center', fontsize=9, transform=ax.get_xaxis_transform())
    ax.text(1, -0.1, f'p={p4:.2e}', ha='center', fontsize=9, transform=ax.get_xaxis_transform())
    
    plt.tight_layout()
    plt.savefig(output_dir / 'layer_comparison_overall.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'layer_comparison_overall.png'}")
    plt.close()
    
    # Plot 4: Difference in correlation by feature pair
    fig, ax = plt.subplots(figsize=(10, 5))
    
    diff_r = layer4_r - layer1_r
    colors = [NORD['green'] if d > 0 else NORD['red'] for d in diff_r]
    
    bars = ax.bar(range(len(feature_pairs)), diff_r, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_xticks(range(len(feature_pairs)))
    ax.set_xticklabels(feature_pairs, rotation=45, ha='right')
    ax.set_ylabel('Δr (Layer -4 minus Layer -1)')
    ax.set_title('Change in VCR-Intervention Correlation\n(Positive = Layer -4 better)')
    ax.axhline(0, color='gray', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'layer_comparison_diff.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'layer_comparison_diff.png'}")
    plt.close()
    
    return corr_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=10, help='Bootstrap seeds per condition')
    parser.add_argument('--feature_pairs', type=str, nargs='+', 
                       default=list(FEATURE_PAIRS.keys()),
                       help='Which feature pairs to test')
    parser.add_argument('--output_dir', type=str, default='./multi_feature_results_v2')
    parser.add_argument('--skip_layer1_run', action='store_true',
                       help='Skip running layer -1 experiment (use existing results)')
    parser.add_argument('--skip_layer4_run', action='store_true',
                       help='Skip running layer -4 experiment (use existing results)')
    parser.add_argument('--model_name', type=str, default='OpenFlamingo-3B-Instruct')
    args = parser.parse_args()
    
    args.output_dir = f'{args.output_dir}_{args.model_name}'
    
    set_nord_style()
    Path(args.output_dir).mkdir(exist_ok=True)
    
    corr_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    # Run layer -1 experiment
    layer1_results_path = Path(args.output_dir) / 'all_results_layer1.csv'
    
    if args.skip_layer1_run and layer1_results_path.exists():
        print("\n" + "="*70)
        print("LOADING EXISTING LAYER -1 RESULTS")
        print("="*70)
        df_layer1 = pd.read_csv(layer1_results_path)
    else:
        print("\n" + "="*70)
        print("RUNNING LAYER -1 EXPERIMENT")
        print("="*70)
        
        all_results_layer1 = []
        
        for feature_pair in args.feature_pairs:
            if feature_pair not in FEATURE_PAIRS:
                print(f"Unknown feature pair: {feature_pair}, skipping")
                continue
            
            df = run_experiment(feature_pair, corr_values, n_seeds=args.n_seeds, 
                              finetune_layer_offset=-1, model_name=args.model_name)
            all_results_layer1.append(df)
            
            # Save intermediate results
            df.to_csv(Path(args.output_dir) / f'results_{feature_pair}_layer1.csv', index=False)
        
        df_layer1 = pd.concat(all_results_layer1, ignore_index=True)
        df_layer1.to_csv(layer1_results_path, index=False)
        print(f"Saved layer -1 results to {layer1_results_path}")
    
    # Run layer -4 experiment
    layer4_results_path = Path(args.output_dir) / 'all_results_layer4.csv'
    
    if args.skip_layer4_run and layer4_results_path.exists():
        print("\n" + "="*70)
        print("LOADING EXISTING LAYER -4 RESULTS")
        print("="*70)
        df_layer4 = pd.read_csv(layer4_results_path)
    else:
        print("\n" + "="*70)
        print("RUNNING LAYER -4 EXPERIMENT")
        print("="*70)
        
        all_results_layer4 = []
        
        for feature_pair in args.feature_pairs:
            if feature_pair not in FEATURE_PAIRS:
                print(f"Unknown feature pair: {feature_pair}, skipping")
                continue
            
            df = run_experiment(feature_pair, corr_values, n_seeds=args.n_seeds, 
                              finetune_layer_offset=-4, model_name=args.model_name)
            all_results_layer4.append(df)
            
            # Save intermediate results
            df.to_csv(Path(args.output_dir) / f'results_{feature_pair}_layer4.csv', index=False)
        
        df_layer4 = pd.concat(all_results_layer4, ignore_index=True)
        df_layer4.to_csv(layer4_results_path, index=False)
        print(f"Saved layer -4 results to {layer4_results_path}")


if __name__ == '__main__':
    main()