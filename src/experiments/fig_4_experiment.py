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
        draw.ellipse([x, y, x+obj_size, y+obj_size], fill=None, outline=color, width=5)
    
    if has_b:  # filled/solid
        x = np.random.randint(size//2 + 10, size - obj_size - 20)
        y = np.random.randint(40, size - obj_size - 40)
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
# ADVERSARIAL DATASET CREATION
# =============================================================================

def create_adversarial_dataset(output_dir, feature_pair, a_corr=1.0, b_train_corr=-0.8, 
                                n_train=400, n_test=200):
    """
    Create adversarial dataset where:
    - Feature A: positively correlated with label in BOTH train and test (reliable)
    - Feature B: NEGATIVELY correlated in TRAIN, POSITIVELY correlated in TEST (spurious!)
    
    This proves feature B is spurious: model shouldn't learn to use it.
    VCR should correctly identify A as important, B as unimportant.
    CLIP-only will be confused by B's positive test correlation.
    
    Args:
        output_dir: Directory to save images and metadata
        feature_pair: Which feature pair to use
        a_corr: Feature A correlation with label (default 1.0 = perfect)
        b_train_corr: Feature B correlation in TRAINING (default -0.8 = strong negative)
                      In TEST, this will be FLIPPED to positive!
        n_train: Number of training examples
        n_test: Number of test examples
    """
    output_dir = Path(output_dir)
    (output_dir / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'test').mkdir(parents=True, exist_ok=True)
    
    generator = GENERATORS[feature_pair]
    pair_info = FEATURE_PAIRS[feature_pair]
    
    n_pos = n_train // 2
    n_neg = n_train // 2
    
    # Feature A: reliable predictor (positive correlation in train)
    n_a_in_pos = int(n_pos * (a_corr + 1) / 2)
    n_a_in_neg = int(n_neg * (1 - (a_corr + 1) / 2))
    
    # Feature B: NEGATIVE correlation in training (will flip in test)
    # If b_train_corr = -0.8, then B appears in 10% of positive, 90% of negative
    n_b_in_pos = int(n_pos * (b_train_corr + 1) / 2)
    n_b_in_neg = int(n_neg * (1 - (b_train_corr + 1) / 2))
    
    train_data = []
    idx = 0
    
    # Training data: B negatively correlated
    for i in range(n_pos):
        has_a = i < n_a_in_pos
        has_b = i < n_b_in_pos  # Few B in positive examples
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
        has_b = i < n_b_in_neg  # Many B in negative examples
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
    
    # Test data: B POSITIVELY correlated (opposite of training!)
    # Labels still determined by A, but B co-occurs with positive examples
    test_data = []
    idx = 0
    
    # Positive examples (has_a=True): 80% have B
    n_test_pos = n_test // 2
    for i in range(n_test_pos):
        has_b = i < int(0.8 * n_test_pos)  # 80% with B
        img = generator(True, has_b, seed=idx+10000)
        filename = f'test_{idx:04d}.png'
        img.save(output_dir / 'test' / filename)
        test_data.append({
            'filename': filename, 
            'has_a': True,
            'has_b': has_b, 
            'label': 'positive'
        })
        idx += 1
    
    # Negative examples (has_a=False): 20% have B
    n_test_neg = n_test // 2
    for i in range(n_test_neg):
        has_b = i < int(0.2 * n_test_neg)  # 20% with B
        img = generator(False, has_b, seed=idx+10000)
        filename = f'test_{idx:04d}.png'
        img.save(output_dir / 'test' / filename)
        test_data.append({
            'filename': filename, 
            'has_a': False,
            'has_b': has_b, 
            'label': 'negative'
        })
        idx += 1
    
    pd.DataFrame(train_data).to_csv(output_dir / 'train_metadata.csv', index=False)
    pd.DataFrame(test_data).to_csv(output_dir / 'test_metadata.csv', index=False)
    
    # Verify correlations
    df_train = pd.DataFrame(train_data)
    df_test = pd.DataFrame(test_data)
    
    train_a = (df_train[df_train['label']=='positive']['has_a'].mean() -
               df_train[df_train['label']=='negative']['has_a'].mean())
    train_b = (df_train[df_train['label']=='positive']['has_b'].mean() -
               df_train[df_train['label']=='negative']['has_b'].mean())
    
    test_a = (df_test[df_test['label']=='positive']['has_a'].mean() -
              df_test[df_test['label']=='negative']['has_a'].mean())
    test_b = (df_test[df_test['label']=='positive']['has_b'].mean() -
              df_test[df_test['label']=='negative']['has_b'].mean())
    
    print(f"  TRAIN - {pair_info['feature_a']}: {train_a:+.2f} | {pair_info['feature_b']}: {train_b:+.2f}")
    print(f"  TEST  - {pair_info['feature_a']}: {test_a:+.2f} | {pair_info['feature_b']}: {test_b:+.2f} (FLIPPED!)")
    
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


def get_model_predictions(model, test_dataset):
    """Get model's P(positive) for each test image."""
    model.model.eval()
    
    pos_token_id = model.tokenizer.encode(" positive", add_special_tokens=False)[0]
    neg_token_id = model.tokenizer.encode(" negative", add_special_tokens=False)[0]
    
    per_image_probs = []
    
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            item = test_dataset[idx]
            
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
            
            per_image_probs.append(pos_prob_normalized)
            
            del image, outputs
    
    return np.array(per_image_probs)


def run_vcr(model, model_name, layer_name, data_dir, concept_files, clip):
    """Run VCR analysis and return sensitivities and CLIP scores."""
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
    
    # Get model predictions for CLIP baseline
    per_image_probs = get_model_predictions(model, test_dataset)
    
    # CLIP-only baseline: correlation with model output
    clip_scores = compute_clip_scores(sim_matrix, per_image_probs)
    
    return concept_texts, raw_sens, clip_scores, sim_matrix


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

def run_experiment(feature_pair, n_seeds=10, model_name='OpenFlamingo-3B-Instruct',
                   finetune_layer_offset=-1, a_corr=1.0, b_train_corr=-0.8):
    """Run adversarial experiment for one feature pair."""
    
    pair_info = FEATURE_PAIRS[feature_pair]
    data_dir = f'./adversarial_data_{feature_pair}'
    if model_name == 'OpenFlamingo-3B-Instruct':
        layer_name = 'model.lang_encoder.transformer.blocks.23.decoder_layer'
    else:
        layer_name = 'model.lang_encoder.gpt_neox.layers.31.decoder_layer'
    concept_files = ['/home/joseph/vlm-interp/src/concept_sets/google-10000-english-no-swears.txt']
    
    print(f"\n{'='*70}")
    print(f"FEATURE PAIR: {feature_pair} (ADVERSARIAL SETUP)")
    print(f"  Feature A: {pair_info['feature_a']} (reliable) -> VCR concept: {pair_info['concept_a']}")
    print(f"  Feature B: {pair_info['feature_b']} (spurious) -> VCR concept: {pair_info['concept_b']}")
    print(f"  Fine-tuning layer offset: {finetune_layer_offset}")
    print(f"  A correlation: {a_corr:.2f} (same in train & test)")
    print(f"  B correlation: {b_train_corr:.2f} in TRAIN, FLIPPED in TEST")
    print(f"{'='*70}")
    
    clip = CLIPEmbedder()
    
    # Create adversarial dataset
    print("\nGenerating adversarial dataset...")
    create_adversarial_dataset(data_dir, feature_pair, a_corr, b_train_corr)
    
    train_df = pd.read_csv(Path(data_dir) / 'train_metadata.csv')
    test_df = pd.read_csv(Path(data_dir) / 'test_metadata.csv')
    
    all_results = []
    concept_texts = None
    
    for seed in tqdm(range(n_seeds), desc=f"  Bootstrap seeds", leave=True):
        model = FlamingoAPI(model_name)
        
        train_resampled = train_df.sample(n=len(train_df), replace=True, random_state=seed)
        train_dataset = SyntheticDataset(train_resampled, Path(data_dir) / 'train', model.image_processor)
        
        model = finetune(model, train_dataset, finetune_layer_offset=finetune_layer_offset)
        
        seed_concepts, raw_sens, clip_scores, sim_matrix = run_vcr(
            model, model_name, layer_name, data_dir, concept_files, clip
        )
        
        if concept_texts is None:
            concept_texts = seed_concepts
        
        idx_a = find_concept_idx(concept_texts, pair_info['concept_a'])
        idx_b = find_concept_idx(concept_texts, pair_info['concept_b'])
        
        if idx_a is not None:
            all_results.append({
                'feature_pair': feature_pair,
                'feature': pair_info['feature_a'],
                'feature_type': 'reliable',
                'vcr_sensitivity': raw_sens.mean(axis=0)[idx_a],
                'clip_score': clip_scores[idx_a],
                'seed': seed,
                'finetune_layer': finetune_layer_offset
            })
        
        if idx_b is not None:
            all_results.append({
                'feature_pair': feature_pair,
                'feature': pair_info['feature_b'],
                'feature_type': 'spurious',
                'vcr_sensitivity': raw_sens.mean(axis=0)[idx_b],
                'clip_score': clip_scores[idx_b],
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

def plot_adversarial_results(all_df, output_dir, layer_label):
    """Generate plots for adversarial experiment results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    feature_pairs = all_df['feature_pair'].unique()
    
    # Plot 1: VCR correctly distinguishes reliable vs spurious
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # VCR results
    ax = axes[0]
    reliable_vcr = all_df[all_df['feature_type'] == 'reliable']['vcr_sensitivity']
    spurious_vcr = all_df[all_df['feature_type'] == 'spurious']['vcr_sensitivity']
    
    positions = [1, 2]
    bp = ax.boxplot([reliable_vcr, spurious_vcr], positions=positions, widths=0.6, 
                    patch_artist=True)
    bp['boxes'][0].set_facecolor(NORD['green'])
    bp['boxes'][1].set_facecolor(NORD['red'])
    
    ax.set_xticks(positions)
    ax.set_xticklabels(['Reliable\n(A)', 'Spurious\n(B)'])
    ax.set_ylabel('VCR Sensitivity')
    ax.set_title(f'VCR (gradient-based)\nLayer {layer_label}')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    
    # Add significance test
    t_stat, p_val = stats.ttest_ind(reliable_vcr, spurious_vcr)
    ax.text(0.5, 0.95, f't={t_stat:.2f}, p={p_val:.2e}', transform=ax.transAxes, 
            ha='center', fontsize=10)
    
    # CLIP results
    ax = axes[1]
    reliable_clip = all_df[all_df['feature_type'] == 'reliable']['clip_score']
    spurious_clip = all_df[all_df['feature_type'] == 'spurious']['clip_score']
    
    bp = ax.boxplot([reliable_clip, spurious_clip], positions=positions, widths=0.6,
                    patch_artist=True)
    bp['boxes'][0].set_facecolor(NORD['green'])
    bp['boxes'][1].set_facecolor(NORD['red'])
    
    ax.set_xticks(positions)
    ax.set_xticklabels(['Reliable\n(A)', 'Spurious\n(B)'])
    ax.set_ylabel('CLIP Correlation with Output')
    ax.set_title(f'CLIP-only (no gradients)\nLayer {layer_label}')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    
    t_stat_clip, p_val_clip = stats.ttest_ind(reliable_clip, spurious_clip)
    ax.text(0.5, 0.95, f't={t_stat_clip:.2f}, p={p_val_clip:.2e}', transform=ax.transAxes,
            ha='center', fontsize=10)
    
    plt.suptitle('Adversarial Test: VCR vs CLIP-only\n(Feature B spuriously correlated in test set)', 
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f'adversarial_vcr_vs_clip_layer{layer_label}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / f'adversarial_vcr_vs_clip_layer{layer_label}.png'}")
    plt.close()
    
    # Plot 2: Per-feature-pair breakdown
    n_pairs = len(feature_pairs)
    fig, axes = plt.subplots(2, (n_pairs + 1) // 2, figsize=(4 * ((n_pairs + 1) // 2), 8))
    axes = axes.flatten()
    
    for i, fp in enumerate(feature_pairs):
        ax = axes[i]
        fp_data = all_df[all_df['feature_pair'] == fp]
        pair_info = FEATURE_PAIRS[fp]
        
        # Get data
        a_vcr = fp_data[fp_data['feature_type'] == 'reliable']['vcr_sensitivity'].values
        b_vcr = fp_data[fp_data['feature_type'] == 'spurious']['vcr_sensitivity'].values
        a_clip = fp_data[fp_data['feature_type'] == 'reliable']['clip_score'].values
        b_clip = fp_data[fp_data['feature_type'] == 'spurious']['clip_score'].values
        
        x = np.arange(2)
        width = 0.35
        
        ax.bar(x - width/2, [a_vcr.mean(), b_vcr.mean()], width, 
               yerr=[a_vcr.std(), b_vcr.std()], label='VCR', color=NORD['blue'], alpha=0.8, capsize=3)
        ax.bar(x + width/2, [a_clip.mean(), b_clip.mean()], width,
               yerr=[a_clip.std(), b_clip.std()], label='CLIP', color=NORD['orange'], alpha=0.8, capsize=3)
        
        ax.set_xticks(x)
        ax.set_xticklabels([f'{pair_info["feature_a"]}\n(reliable)', 
                           f'{pair_info["feature_b"]}\n(spurious)'])
        ax.set_title(fp)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.legend(fontsize=8)
    
    # Hide extra axes
    for i in range(len(feature_pairs), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Per-Feature-Pair Results (Layer {layer_label})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f'adversarial_per_feature_layer{layer_label}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / f'adversarial_per_feature_layer{layer_label}.png'}")
    plt.close()
    
    # Plot 3: Summary bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate mean difference (reliable - spurious) for each method
    vcr_diffs = []
    clip_diffs = []
    
    for fp in feature_pairs:
        fp_data = all_df[all_df['feature_pair'] == fp]
        vcr_reliable = fp_data[fp_data['feature_type'] == 'reliable']['vcr_sensitivity'].mean()
        vcr_spurious = fp_data[fp_data['feature_type'] == 'spurious']['vcr_sensitivity'].mean()
        clip_reliable = fp_data[fp_data['feature_type'] == 'reliable']['clip_score'].mean()
        clip_spurious = fp_data[fp_data['feature_type'] == 'spurious']['clip_score'].mean()
        
        vcr_diffs.append(vcr_reliable - vcr_spurious)
        clip_diffs.append(clip_reliable - clip_spurious)
    
    x = np.arange(len(feature_pairs))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, vcr_diffs, width, label='VCR', color=NORD['blue'], alpha=0.8)
    bars2 = ax.bar(x + width/2, clip_diffs, width, label='CLIP-only', color=NORD['orange'], alpha=0.8)
    
    ax.set_ylabel('Δ Score (Reliable - Spurious)')
    ax.set_title('Method Discrimination: Reliable vs Spurious Features\n(Positive = correctly ranks reliable > spurious)')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_pairs, rotation=45, ha='right')
    ax.legend()
    ax.axhline(0, color='gray', linestyle='-', linewidth=1)
    
    # Mark successes/failures
    for i, (v, c) in enumerate(zip(vcr_diffs, clip_diffs)):
        marker = '✓' if v > 0 else '✗'
        ax.text(i - width/2, v + 0.01 if v > 0 else v - 0.03, marker, ha='center', fontsize=10)
        marker = '✓' if c > 0 else '✗'
        ax.text(i + width/2, c + 0.01 if c > 0 else c - 0.03, marker, ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'adversarial_discrimination_layer{layer_label}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / f'adversarial_discrimination_layer{layer_label}.png'}")
    plt.close()
    
    return vcr_diffs, clip_diffs


def plot_layer_comparison(df_layer1, df_layer4, output_dir):
    """Compare adversarial results between fine-tuning layers."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, (layer_name, layer_df) in zip(axes, [('Layer -1', df_layer1), ('Layer -4', df_layer4)]):
        reliable_vcr = layer_df[layer_df['feature_type'] == 'reliable']['vcr_sensitivity']
        spurious_vcr = layer_df[layer_df['feature_type'] == 'spurious']['vcr_sensitivity']
        
        positions = [1, 2]
        bp = ax.boxplot([reliable_vcr, spurious_vcr], positions=positions, widths=0.6,
                        patch_artist=True)
        bp['boxes'][0].set_facecolor(NORD['green'])
        bp['boxes'][1].set_facecolor(NORD['red'])
        
        ax.set_xticks(positions)
        ax.set_xticklabels(['Reliable', 'Spurious'])
        ax.set_ylabel('VCR Sensitivity')
        ax.set_title(f'{layer_name}')
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        
        t_stat, p_val = stats.ttest_ind(reliable_vcr, spurious_vcr)
        discrimination = reliable_vcr.mean() - spurious_vcr.mean()
        ax.text(0.5, 0.95, f'Δ={discrimination:.3f}\np={p_val:.2e}', transform=ax.transAxes,
                ha='center', fontsize=10)
    
    plt.suptitle('VCR Discrimination by Fine-tuning Layer', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'layer_comparison_adversarial.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'layer_comparison_adversarial.png'}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=10, help='Bootstrap seeds per condition')
    parser.add_argument('--feature_pairs', type=str, nargs='+', 
                       default=list(FEATURE_PAIRS.keys()),
                       help='Which feature pairs to test')
    parser.add_argument('--output_dir', type=str, default='./adversarial_multifeature_v2')
    parser.add_argument('--skip_layer1_run', action='store_true',
                       help='Skip running layer -1 experiment (use existing results)')
    parser.add_argument('--skip_layer4_run', action='store_true',
                       help='Skip running layer -4 experiment (use existing results)')
    parser.add_argument('--model_name', type=str, default='OpenFlamingo-4B')
    parser.add_argument('--a_corr', type=float, default=1.0,
                       help='Feature A correlation (reliable, same in train & test)')
    parser.add_argument('--b_train_corr', type=float, default=-0.8,
                       help='Feature B correlation in TRAINING (flipped in test)')
    args = parser.parse_args()
    
    args.output_dir = f'{args.output_dir}_{args.model_name}'
    
    set_nord_style()
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Run layer -1 experiment
    layer1_results_path = Path(args.output_dir) / 'all_results_layer1.csv'
    
    if args.skip_layer1_run and layer1_results_path.exists():
        print("\n" + "="*70)
        print("LOADING EXISTING LAYER -1 RESULTS")
        print("="*70)
        df_layer1 = pd.read_csv(layer1_results_path)
    else:
        print("\n" + "="*70)
        print("RUNNING LAYER -1 EXPERIMENT (ADVERSARIAL)")
        print("="*70)
        
        all_results_layer1 = []
        
        for feature_pair in args.feature_pairs:
            if feature_pair not in FEATURE_PAIRS:
                print(f"Unknown feature pair: {feature_pair}, skipping")
                continue
            
            df = run_experiment(
                feature_pair, 
                n_seeds=args.n_seeds, 
                model_name=args.model_name,
                finetune_layer_offset=-1,
                a_corr=args.a_corr,
                b_train_corr=args.b_train_corr
            )
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
        print("RUNNING LAYER -4 EXPERIMENT (ADVERSARIAL)")
        print("="*70)
        
        all_results_layer4 = []
        
        for feature_pair in args.feature_pairs:
            if feature_pair not in FEATURE_PAIRS:
                print(f"Unknown feature pair: {feature_pair}, skipping")
                continue
            
            df = run_experiment(
                feature_pair, 
                n_seeds=args.n_seeds, 
                model_name=args.model_name,
                finetune_layer_offset=-4,
                a_corr=args.a_corr,
                b_train_corr=args.b_train_corr
            )
            all_results_layer4.append(df)
            
            # Save intermediate results
            df.to_csv(Path(args.output_dir) / f'results_{feature_pair}_layer4.csv', index=False)
        
        df_layer4 = pd.concat(all_results_layer4, ignore_index=True)
        df_layer4.to_csv(layer4_results_path, index=False)
        print(f"Saved layer -4 results to {layer4_results_path}")


if __name__ == '__main__':
    main()