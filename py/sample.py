#!/usr/bin/env python
"""
sample_simple.py - Sampling from models trained with train_simple_v2.py
Works with both standard softmax attention and linear attention models.
Supports both character-level and BPE tokenization.

Usage: python sample_simple_v2.py --model model.pt --prompt "The Roman" --rep_penalty 1.15
"""

import os
import sys
import argparse
import pickle
import torch
from contextlib import nullcontext
from model import GPTConfig, GPT
from tokenizer import load_tokenizer


def apply_repetition_penalty(logits, generated_ids, penalty=1.2, window=50):
    """
    Reduce repetition by penalizing recently used tokens.

    Args:
        logits: Raw model outputs [vocab_size]
        generated_ids: List of token IDs generated so far
        penalty: Multiplicative penalty (>1.0 suppresses repetition)
        window: How many recent tokens to consider
    """
    # Look at recent tokens (or all if sequence is short)
    lookback = min(len(generated_ids), window)
    if lookback == 0:
        return logits

    recent_tokens = generated_ids[-lookback:]

    # Penalize each recent token proportional to its frequency
    for token_id in set(recent_tokens):
        count = recent_tokens.count(token_id)
        # Apply graduated penalty based on frequency
        logits[token_id] = logits[token_id] / (penalty ** count)

    return logits


@torch.no_grad()
def generate_local(model, x_init, max_new_tokens, temperature=1.0, top_k=None, rep_penalty=1.0, device='cpu', stop_token_id=None):
    """
    Generate text from initial prompt with repetition penalty
    x_init: (1, T) prompt indices
    stop_token_id: Optional token ID to stop generation (e.g., newline)
    Returns: (1, T+max_new_tokens)
    """
    x = x_init
    block_size = model.config.block_size
    generated_ids = []  # Track generated tokens for repetition penalty

    for _ in range(max_new_tokens):
        # Crop context if needed
        if x.size(1) > block_size:
            idx_cond = x[:, -block_size:]
        else:
            idx_cond = x

        # Get model prediction
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # (1, vocab_size)

        # Apply repetition penalty if enabled
        if rep_penalty > 1.0 and len(generated_ids) > 0:
            logits = apply_repetition_penalty(
                logits.squeeze(0),
                generated_ids,
                penalty=rep_penalty,
                window=500
            ).unsqueeze(0)

        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = -float('inf')

        # Apply temperature and sample
        if temperature <= 0.0:
            # Greedy decoding
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # Sample with temperature
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        # Track generated token for repetition penalty
        generated_ids.append(idx_next.item())
        
        # Check for stop token
        if stop_token_id is not None and idx_next.item() == stop_token_id:
            break
            
        x = torch.cat((x, idx_next), dim=1)

    return x


def main():
    parser = argparse.ArgumentParser(description='Sample from character-level GPT (supports linear attention and BPE)')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint (.pt file)')
    parser.add_argument('--prompt', type=str, default="\n", help='Starting prompt text')
    parser.add_argument('--prompt_file', type=str, help='File containing prompt text')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to generate')
    parser.add_argument('--max_tokens', type=int, default=130, help='Maximum new tokens per sample')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature (0=greedy)')
    parser.add_argument('--top_k', type=int, default=40, help='Top-k filtering (0=disabled)')
    parser.add_argument('--rep_penalty', type=float, default=0.0,
                        help='Repetition penalty 0.0=off, 1.15=gentle, 1.3=aggressive')
    parser.add_argument('--no_stop_on_newline', action='store_true',
                        help='Continue past newline instead of stopping (default: stop at newline)')
    parser.add_argument('--corpus', type=str, default=None,
                        help='Path to corpus file (one word per line) for validation marking')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (default: None=random each run)')

    args = parser.parse_args()

    # Check model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found")
        sys.exit(1)

    # Determine metadata file path
    # Handle regular, _iter{N}, and _final checkpoint names
    model_base = args.model.replace('.pt', '')
    if '_iter' in model_base:
        # Strip _iter{N} suffix to get base name
        model_base = model_base.rsplit('_iter', 1)[0]
    elif model_base.endswith('_final'):
        # Strip _final suffix
        model_base = model_base[:-6]
    meta_path = model_base + '_meta.pkl'
    if not os.path.exists(meta_path):
        print(f"Error: Metadata file '{meta_path}' not found")
        print("Make sure this model was trained with train_simple_v2.py")
        sys.exit(1)

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        print(f"Using seed: {args.seed}")
    else:
        # Use random seed (from system time/entropy)
        import time
        seed = int(time.time() * 1000) % (2**32)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        print(f"Using random seed: {seed}")

    # Device selection
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)

    # Initialize model
    model_args = checkpoint['model_args']

    # Check attention type
    use_linear = model_args.get('use_linear_attention', False)
    attn_type = "linear" if use_linear else "softmax"
    print(f"Attention type: {attn_type}")

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # Load tokenizer
    print(f"Loading tokenizer from {meta_path}")
    tokenizer = load_tokenizer(meta_path)

    vocab_size = tokenizer.vocab_size
    tokenizer_type = tokenizer.tokenizer_type
    print(f"Tokenizer: {tokenizer_type}")
    print(f"Vocabulary size: {vocab_size} tokens")
    
    # Load corpus for validation if provided
    corpus_words = None
    if args.corpus:
        if not os.path.exists(args.corpus):
            print(f"Warning: Corpus file '{args.corpus}' not found, skipping validation")
        else:
            with open(args.corpus, 'r', encoding='utf-8') as f:
                corpus_words = set(word.strip() for word in f.read().strip().split('\n') if word.strip())
            print(f"Loaded corpus: {len(corpus_words)} unique words")
    
    # Determine newline token ID for stopping (default: stop at newline)
    stop_token_id = None
    if not args.no_stop_on_newline:
        newline_ids = tokenizer.encode('\n')
        if newline_ids:
            stop_token_id = newline_ids[0]
            print(f"Stop token ID (newline): {stop_token_id}")
        else:
            print("Warning: Could not find newline token in vocabulary")

#--------------------------------------------------

    # Get prompt
    if args.prompt_file:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompt_text = f.read()
    else:
        prompt_text = args.prompt

    print(f"\nPrompt: '{prompt_text[:50]}{'...' if len(prompt_text) > 50 else ''}'")
    print(f"Generating {args.num_samples} samples...")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}, Rep-penalty: {args.rep_penalty}")
    print("=" * 70)


    # Encode prompt
    prompt_ids = tokenizer.encode(prompt_text)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]

    # Generate samples
    for i in range(args.num_samples):
        # Generate with repetition penalty
        y = generate_local(model, x, args.max_tokens,
                          temperature=args.temperature,
                          top_k=args.top_k if args.top_k > 0 else None,
                          rep_penalty=args.rep_penalty,
                          device=device,
                          stop_token_id=stop_token_id)

        # Decode and print (one line per sample)
        generated_text = tokenizer.decode(y[0].tolist())
        
        # If corpus validation is enabled and we're generating single words
        if corpus_words is not None and not args.no_stop_on_newline:
            word = generated_text.strip()
            if word in corpus_words:
                generated_text = word + ' *'
            else:
                generated_text = word
        
        print(generated_text)

    print("\n" + "=" * 70)

    # Print some statistics about the generation

#--------------------------------------------------

#     # Encode prompt
#     prompt_ids = tokenizer.encode(prompt_text)
#     x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]
#
#     # Generate samples
#     for i in range(args.num_samples):
#         print(f"\nSample {i+1}:")
#         print("-" * 50)
#
#         # Generate with repetition penalty
#         y = generate_local(model, x, args.max_tokens,
#                           temperature=args.temperature,
#                           top_k=args.top_k if args.top_k > 0 else None,
#                           rep_penalty=args.rep_penalty,
#                           device=device)
#
#         # Decode and print
#         generated_text = tokenizer.decode(y[0].tolist())
#         print(generated_text)
#
#     print("\n" + "=" * 70)
#
#     # Print some statistics about the generation

#--------------------------------------------------

    if checkpoint.get('iter_num'):
        print(f"Model trained for {checkpoint['iter_num']} iterations")
    if checkpoint.get('best_val_loss'):
        print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")


if __name__ == '__main__':
    main()
