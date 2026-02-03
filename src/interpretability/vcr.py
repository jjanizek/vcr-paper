#!/usr/bin/env python
# vcr.py
"""
Core module for visual concept-based analysis of LMMs.
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
from einops import repeat
import re
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
import torch.nn.functional as F
from models.flamingo import FlamingoAPI
from collections import defaultdict

class PromptTemplate:
    """Class for managing prompt templates."""
    
    def __init__(self, base_prompt, demo_template=None, query_template=None):
        """
        Initialize prompt template.
        
        Args:
            base_prompt: Base prompt string (can include {demo} placeholder)
            demo_template: Template for each demonstration
            query_template: Template for the query
        """
        self.base_prompt = base_prompt
        self.demo_template = demo_template or "<image>Based on the image, this lesion is {label}.<|endofchunk|>"
        self.query_template = query_template or "<image>Based on the image, this lesion is"
    
    def build_prompt(self, demo_labels=None):
        """Build the full prompt with optional demonstrations."""
        if demo_labels is None:
            # Zero-shot case
            return self.base_prompt + self.query_template
        else:
            # Few-shot case
            demo_prompts = ""
            for label in demo_labels:
                demo_prompts += self.demo_template.format(label=label)
            return self.base_prompt + demo_prompts + self.query_template


class ConceptAnalyzer:
    """Main class for concept-based model analysis."""
    
    def __init__(self, model_name, clip_embedder, image_processor=None):
        """
        Initialize the analyzer.
        
        Args:
            model: The vision-language model to analyze
            clip_embedder: CLIP embedder for computing concept embeddings
            image_processor: Optional image processor (will use model's if not provided)
        """
        
        self.model = FlamingoAPI(model_name=model_name)
        self.model_name = model_name
        self.clip = clip_embedder
        self.image_processor = image_processor or self.model.image_processor
        self.wrapped_layer = None
        self.concept_model = None
        self.concept_vectors = None
        
    def setup_layer_hook(self, target_layer_name, layer_wrapper_class):
        """
        Set up a hook on the specified layer.

        Args:
            target_layer_name: Dot-separated path to the layer
            layer_wrapper_class: Class to wrap the layer with (e.g., LayerOverride)

        Returns:
            The wrapped layer module
        """
        # Find the target layer
        parts = target_layer_name.split('.')

        # Start from the model object
        target_module = self.model

        # Check if we need to access the inner model first
        # This handles cases where the model is wrapped in FlamingoAPI
        if hasattr(self.model, 'model'):
            # Check if the first part exists in the wrapper or the inner model
            if hasattr(self.model, parts[0]):
                target_module = self.model
            elif hasattr(self.model.model, parts[0]):
                target_module = self.model.model
            else:
                # For debugging, let's see what attributes are available
                print(f"Available attributes in self.model: {dir(self.model)}")
                if hasattr(self.model, 'model'):
                    print(f"Available attributes in self.model.model: {dir(self.model.model)}")
                raise AttributeError(f"Cannot find {parts[0]} in model structure")

        # Extract block/layer number for debugging
        block_num = None
        if self.model_name == 'OpenFlamingo-3B-Instruct': 
            block_match = re.search(r'blocks\.(\d+)', target_layer_name)
            block_num = int(block_match.group(1)) if block_match else None
        elif self.model_name in ['MedFlamingo', 'OpenFlamingo-4B']:
            block_match = re.search(r'layers\.(\d+)', target_layer_name)
            block_num = int(block_match.group(1)) if block_match else None

        # Store the root module for later
        root_module = target_module

        # Traverse to the target module
        for part in parts:
            if part.isdigit():
                target_module = target_module[int(part)]
            else:
                target_module = getattr(target_module, part)

        # Wrap the layer
        self.wrapped_layer = layer_wrapper_class(target_module)

        # Set the wrapped module back in the model
        current_module = root_module

        for i, part in enumerate(parts[:-1]):
            if part.isdigit():
                current_module = current_module[int(part)]
            else:
                current_module = getattr(current_module, part)

        last_part = parts[-1]
        if last_part.isdigit():
            current_module[int(last_part)] = self.wrapped_layer
        else:
            setattr(current_module, last_part, self.wrapped_layer)

        # Print what we've wrapped for debugging
        print(f"Wrapped {target_layer_name} (block {block_num})" if block_num is not None else f"Wrapped {target_layer_name}")

        return self.wrapped_layer

    def get_embeddings(self, image_paths, concept_files):
        """
        Compute CLIP embeddings for images and concepts.
        
        Args:
            image_paths: List of paths to images
            concept_files: List of text files containing concepts
            
        Returns:
            Tuple of (image_embeddings, text_embeddings, concept_texts)
        """
        # Load concept texts
        texts = []
        for f in concept_files:
            with open(f) as file:
                texts.extend(line.strip() for line in file)
        
        # Compute embeddings
        image_embeddings = self.clip.get_image_embeddings(image_paths)
        text_embeddings = self.clip.get_text_embeddings(texts)
        
        return image_embeddings, text_embeddings, texts
    
    def process_images_for_model(self, image_paths):
        """Process a list of images for model input."""
        processed_images = []
        for img_path in image_paths:
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            processed_images.append(self.image_processor(image))
        return processed_images
    
    def collect_activations(self, dataset, prompt_template, demo_paths=None, 
                          demo_labels=None, batch_size=1, num_workers=4):
        """
        Collect activations from the wrapped layer.
        
        Args:
            dataset: Dataset of images to process
            prompt_template: Template for generating prompts
            demo_paths: Optional paths to demonstration images for ICL
            demo_labels: Optional labels for demonstration images
            batch_size: Batch size for processing
            num_workers: Number of dataloader workers
            
        Returns:
            Tensor of collected activations
        """
        if self.wrapped_layer is None:
            raise RuntimeError("No layer wrapped. Call setup_layer_hook first.")
            
        dataloader = DataLoader(dataset, batch_size=batch_size, 
                              num_workers=num_workers, shuffle=False)
        layer_outputs = []
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                layer_outputs.append(output[0].detach().cpu())
            else:
                layer_outputs.append(output.detach().cpu())
        
        hook = self.wrapped_layer.register_forward_hook(hook_fn)
        
        # Process demo images if provided
        stacked_demos = None
        if demo_paths is not None and demo_labels is not None:
            processed_imgs = self.process_images_for_model(demo_paths)
            demo_batch = torch.stack(processed_imgs)
            stacked_demos = repeat(demo_batch, "d c h w -> b d 1 c h w", b=batch_size)
        
        # Build prompt
        prompt = prompt_template.build_prompt(demo_labels if demo_labels else None)
        batch_prompt = [prompt] * batch_size
        
        self.model.model.eval()
        for batch in tqdm(dataloader, desc="Collecting activations"):
            image_batch = batch['image'].cuda()
            if len(image_batch.shape) == 4:
                image_batch = image_batch.unsqueeze(1).unsqueeze(2)
            
            if stacked_demos is not None:
                image_batch = torch.cat([stacked_demos.cuda(), image_batch], axis=1)
            
            encoded = self.model.tokenizer(batch_prompt, return_tensors="pt", 
                                         padding=True, truncation=True)
            input_ids = encoded["input_ids"].cuda()
            attention_mask = encoded["attention_mask"].cuda()
            
            outputs = self.model.model(vision_x=image_batch, lang_x=input_ids, 
                                     attention_mask=attention_mask)
        
        hook.remove()
        activations = torch.cat(layer_outputs, dim=0)
        # Add this debug code in collect_activations:
        print(f"Layer activation shape: {activations.shape}")
        print(f"Memory usage: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        return activations
    
    def train_concept_model(self, activations, similarity_matrix, 
                          test_size=0.2, alpha=1.0, random_state=42):
        """
        Train a model to predict concept similarities from activations.
        
        Args:
            activations: Tensor of layer activations [samples, features]
            similarity_matrix: Tensor of concept similarities [concepts, samples]
            test_size: Proportion of data to use for testing
            alpha: Ridge regression regularization parameter
            random_state: Random seed for train/test split
            
        Returns:
            Dictionary containing model, predictions, and metrics
        """
        from sklearn.model_selection import train_test_split
        
        # Use only the last token's activations
        if len(activations.shape) == 3:
            X = activations[:, -1, :].numpy()
        else:
            X = activations.numpy()
            
        Y = similarity_matrix.T.numpy()  # Transpose to [samples, concepts]
        
        # Split data
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state
        )
        
        # Train model
        self.concept_model = Ridge(alpha=alpha)
        self.concept_model.fit(X_train, Y_train)
        
        # Evaluate
        Y_pred = self.concept_model.predict(X_test)
        
        # Calculate R² for each concept
        r2_per_concept = np.array([
            r2_score(Y_test[:, i], Y_pred[:, i]) 
            for i in range(Y_test.shape[1])
        ])
        
        return {
            'model': self.concept_model,
            'predictions': Y_pred,
            'y_test': Y_test,
            'r2_scores': r2_per_concept,
            'overall_r2': np.mean(r2_per_concept)
        }
        
    def extract_concept_vectors(self):
        """Extract and normalize concept vectors from the trained model."""
        if self.concept_model is None:
            raise RuntimeError("No concept model trained. Call train_concept_model first.")

        # Ridge with multiple outputs stores coefficients as a matrix
        coef_matrix = self.concept_model.coef_

        # For Ridge with multiple outputs, coef_ has shape (n_targets, n_features)
        if len(coef_matrix.shape) == 1:
            # Single output case
            concept_vectors = coef_matrix.reshape(1, -1)
        else:
            # Multiple output case: coef_matrix is already (n_concepts, n_features)
            concept_vectors = coef_matrix

        # Normalize each concept vector to unit length
        norms = np.linalg.norm(concept_vectors, axis=1, keepdims=True)
        concept_vectors = concept_vectors / norms

        self.concept_vectors = torch.tensor(concept_vectors, dtype=torch.float32)
        return self.concept_vectors
    
    def compute_concept_weights(self, similarity_matrix, weight_type='variance'):
        """
        Compute importance weights for concepts.
        
        Args:
            similarity_matrix: Tensor of concept similarities [concepts, samples]
            weight_type: Type of weighting ('variance', 'uniform', etc.)
            
        Returns:
            Tensor of concept weights
        """
        Y = similarity_matrix.T.numpy()  # [samples, concepts]
        
        if weight_type == 'variance':
            weights = np.var(Y, axis=0)
        elif weight_type == 'uniform':
            weights = np.ones(Y.shape[1])
        else:
            raise ValueError(f"Unknown weight type: {weight_type}")
            
        return torch.tensor(weights, dtype=torch.float32)

    def compute_model_outputs(self, image_batch, prompt_batch, completion):
        """
        Compute the difference in log probabilities between choices.
        
        Args:
            image_batch: Tensor of images
            prompt_batch: List of prompts
            completion: Single completion string
            
        Returns:
            Tensor of choice differences
        """
        device = next(self.model.model.parameters()).device
        batch_size = len(prompt_batch)
        
        # Tokenize choices
        choice_ids = self.model.tokenizer.encode(completion, add_special_tokens=False) 
        
        # Initialize log probabilities
        choice_logprobs = torch.zeros(batch_size, 1, device=device)
        
        # Compute log probabilities for the completion
        full_texts = [f"{prompt}{completion}" for prompt in prompt_batch]
            
        # Tokenize
        encoded = self.model.tokenizer(
            full_texts, 
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
            
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
            
        # Get model outputs
        with torch.set_grad_enabled(True):
            outputs = self.model.model(
                vision_x=image_batch,
                lang_x=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            
        # Calculate log probabilities
        for i in range(batch_size):
            prompt_tokens = self.model.tokenizer.encode(
                prompt_batch[i], add_special_tokens=True
            )
            choice_start = len(prompt_tokens) - 1

            log_prob = 0
            for j, token_id in enumerate(choice_ids):
                pos = choice_start + j
                if pos >= logits.shape[1]:
                    break

                token_logits = logits[i, pos]
                token_log_probs = F.log_softmax(token_logits, dim=-1)
                log_prob += token_log_probs[token_id]

            # Length-normalized log probability
            choice_logprobs[i, 0] = log_prob / max(1, len(choice_ids))
        
        # Return difference between second and first choice
        return choice_logprobs[:, 0]
    
    def calculate_directional_derivatives(self, dataset, concept_vectors, 
                                        concept_weights, prompt_template,
                                        completion, demo_paths=None, demo_labels=None):
        """
        Calculate weighted directional derivatives for concepts.
        
        Args:
            dataset: Dataset to analyze
            concept_vectors: Tensor of concept direction vectors
            concept_weights: Tensor of concept importance weights
            prompt_template: Template for generating prompts
            completion: string of completion
            demo_paths: Optional demonstration image paths
            demo_labels: Optional demonstration labels
            
        Returns:
            Tuple of (weighted_sensitivities, raw_sensitivities)
        """
        if self.wrapped_layer is None:
            raise RuntimeError("No layer wrapped. Call setup_layer_hook first.")
            
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        concept_vectors = concept_vectors.cuda()
        concept_weights = concept_weights.cuda()
        
        # Process demos if provided
        stacked_demos = None
        if demo_paths is not None and demo_labels is not None:
            processed_imgs = self.process_images_for_model(demo_paths)
            demo_batch = torch.stack(processed_imgs)
            stacked_demos = repeat(demo_batch, "d c h w -> b d 1 c h w", b=1)
        
        # Build prompt
        prompt = prompt_template.build_prompt(demo_labels if demo_labels else None)
        
        # Get final token position
        encoded = self.model.tokenizer(prompt, return_tensors="pt")
        final_tok_position = encoded["input_ids"].shape[1] - 1
        
        all_raw_sensitivities = []
        all_weighted_sensitivities = []
        
        for batch in tqdm(dataloader, desc="Calculating sensitivities"):
            torch.cuda.empty_cache()
            
            # Hook to collect layer outputs
            layer_outputs = []
            def hook_fn(module, input, output):
                layer_outputs.append(output)
            hook = self.wrapped_layer.register_forward_hook(hook_fn)
            
            image_batch = batch['image'].cuda()
            if len(image_batch.shape) == 4:
                image_batch = image_batch.unsqueeze(1).unsqueeze(2)
            
            if stacked_demos is not None:
                image_batch = torch.cat([stacked_demos.cuda(), image_batch], axis=1)
            
            prompt_batch = [prompt]
            
            # Compute choice difference
            outputs = self.compute_model_outputs(
                image_batch, prompt_batch, completion
            )
            
            # Compute gradient
            activation_grad = torch.autograd.grad(
                outputs=outputs,
                inputs=layer_outputs[-1][0],
                create_graph=False,
                retain_graph=False
            )[0]
            
            hook.remove()
            
            # Flatten gradient
            flattened_grad = activation_grad.view(activation_grad.size(1), -1)
            
            # Compute directional derivatives
            raw_sensitivities = torch.matmul(flattened_grad, concept_vectors.T)
            weighted_sensitivities = raw_sensitivities * concept_weights.unsqueeze(0)
            
            # Store results for final token
            all_raw_sensitivities.append(
                raw_sensitivities.cpu().detach().numpy()[final_tok_position, :]
            )
            all_weighted_sensitivities.append(
                weighted_sensitivities.cpu().detach().numpy()[final_tok_position, :]
            )
            
            # At the end of each batch:
            del activation_grad, raw_sensitivities, weighted_sensitivities
            del layer_outputs[:]  # Clear the list
            torch.cuda.empty_cache()
        
        # Stack results
        all_weighted = np.vstack(all_weighted_sensitivities)
        all_raw = np.vstack(all_raw_sensitivities)
        
        return all_weighted, all_raw