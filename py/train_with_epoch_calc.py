#!/usr/bin/env python3
"""
Training script with optional linear attention and tokenizer choice.
Based on train_simple.py but adds support for both character and BPE tokenization.

Usage:
  Character (default): python train_simple_v2.py --input data.txt
  BPE tokenization:    python train_simple_v2.py --input data.txt --tokenizer bpe --vocab_size 16384
  Resume training:     python train_simple_v2.py --input data.txt --resume pt/model_iter2000.pt
"""

import argparse
import os
import sys
import time
import math
import pickle
import numpy as np
import torch
from datetime import datetime
from contextlib import nullcontext
from model import GPT, GPTConfig
from tokenizer import CharTokenizer, BPETokenizer, load_tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description='Train GPT model with optional linear attention and tokenizer choice')

    # Required arguments
    parser.add_argument('--input', type=str, required=True,
                       help='Input text file for training')

    # Optional arguments with defaults
    parser.add_argument('--output', type=str, default=None,
                       help='Output model file name (default: based on input name)')
    parser.add_argument('--n_layer', type=int, default=12,
                       help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=12,
                       help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=768,
                       help='Embedding dimension')
    parser.add_argument('--block_size', type=int, default=1024,
                       help='Context length')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--max_iters', type=int, default=100000,
                       help='Maximum training iterations')
    parser.add_argument('--eval_interval', type=int, default=500,
                       help='How often to evaluate')
    parser.add_argument('--eval_iters', type=int, default=20,
                       help='Number of iterations for evaluation')
    parser.add_argument('--learning_rate', type=float, default=6e-4,
                       help='Learning rate')
    parser.add_argument('--warmup_iters', type=int, default=2000,
                       help='Warmup iterations')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--save_interval', type=int, default=10000,
                       help='Checkpoint save interval')
    parser.add_argument('--max_tokens_in_block', type=int, default=None,
                       help='Maximum word length in tokens (must be <= block_size). Words exceeding this are skipped.')

    # Linear attention option
    parser.add_argument('--linear_attention', action='store_true',
                       help='Use linear attention instead of softmax attention')

    # Weight tying option
    parser.add_argument('--untie_weights', action='store_true',
                       help='Do not tie input embeddings to output projection')

    # Precision option
    parser.add_argument('--precision', type=str, default='float32',
                       choices=['float16', 'float32', 'float64', 'bfloat16'],
                       help='Floating-point precision (default: float32)')

    # Tokenizer options
    parser.add_argument('--tokenizer', type=str, default='char',
                       choices=['char', 'bpe'],
                       help='Tokenization scheme: char (default) or bpe')
    parser.add_argument('--vocab_size', type=int, default=16384,
                       help='Vocabulary size for BPE tokenizer (default: 16384, ignored for char)')

    # Resume option
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')

    return parser.parse_args()


def get_timestamp():
    """Get current timestamp in readable format"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate_epoch(iter_num, batch_size, block_size, train_tokens):
    """Calculate approximate epoch from iteration number"""
    tokens_per_iter = batch_size * block_size
    tokens_processed = iter_num * tokens_per_iter
    epoch = tokens_processed / train_tokens
    return epoch


def get_batch(data, batch_size, block_size, device='cpu'):
    """
    Generate a batch of data for training from pre-blocked data.
    data shape: (num_blocks, block_size)
    """
    # Sample random blocks
    num_blocks = len(data)
    ix = torch.randint(num_blocks, (batch_size,))

    # Extract blocks - data is already (num_blocks, block_size)
    x = torch.from_numpy(data[ix, :-1].astype(np.int64))  # All but last token
    y = torch.from_numpy(data[ix, 1:].astype(np.int64))   # All but first token

    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters, device):
    """Estimate loss on train and validation sets"""
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(iter_num, learning_rate, warmup_iters, max_iters):
    """Learning rate schedule with linear warmup and cosine decay"""
    min_lr = learning_rate / 10
    # Linear warmup
    if iter_num < warmup_iters:
        return learning_rate * iter_num / warmup_iters
    # Cosine decay
    if iter_num > max_iters:
        return min_lr
    decay_ratio = (iter_num - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def print_vocabulary(tokenizer):
    """Print the complete vocabulary mapping"""
    print(f"\n[{get_timestamp()}] Vocabulary ({tokenizer.vocab_size} tokens):")
    print("=" * 60)

    if tokenizer.tokenizer_type == 'char':
        # For character tokenizer, show character -> ID mapping
        items = sorted(tokenizer.stoi.items(), key=lambda x: x[1])
        for char, idx in items:
            # Make special characters visible
            if char == ' ':
                display = '·'  # middot for space
            elif char == '\n':
                display = '\\n'
            elif char == '\t':
                display = '\\t'
            else:
                display = char
            print(f"  {idx:3d}: '{display}'")
    elif tokenizer.tokenizer_type == 'bpe':
        # For BPE tokenizer, show token ID -> string mapping
        vocab = tokenizer.tokenizer.get_vocab()
        # Sort by token ID
        items = sorted(vocab.items(), key=lambda x: x[1])
        for token, idx in items:
            # Make special characters visible
            display = token.replace(' ', '·').replace('\n', '\\n').replace('\t', '\\t')
            print(f"  {idx:3d}: '{display}'")

    print("=" * 60)


def main():
    args = parse_args()

    # Validate tokenizer arguments (only when not resuming)
    if not args.resume and args.tokenizer == 'char' and args.vocab_size != 16384:
        # User specified vocab_size with char tokenizer
        print("Error: --vocab_size cannot be specified with --tokenizer char (vocabulary is determined from data)")
        sys.exit(1)

    # Validate max_tokens_in_block
    if args.max_tokens_in_block is not None:
        if args.max_tokens_in_block > args.block_size:
            print(f"Error: --max_tokens_in_block ({args.max_tokens_in_block}) cannot exceed --block_size ({args.block_size})")
            sys.exit(1)
    else:
        # Default to block_size if not specified
        args.max_tokens_in_block = args.block_size

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"[{get_timestamp()}] Using device: {device}")

    # Set up precision
    precision_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
        'bfloat16': torch.bfloat16,
    }
    ptdtype = precision_map[args.precision]

    # Check device compatibility and adjust if needed
    original_precision = args.precision
    if device == 'mps':
        if args.precision == 'float64':
            print(f"[{get_timestamp()}] WARNING: MPS does not support float64. Falling back to CPU.")
            device = 'cpu'
        elif args.precision == 'bfloat16':
            print(f"[{get_timestamp()}] WARNING: MPS does not support bfloat16. Falling back to float32.")
            args.precision = 'float32'
            ptdtype = torch.float32
    elif device == 'cuda':
        if args.precision == 'bfloat16' and not torch.cuda.is_bf16_supported():
            print(f"[{get_timestamp()}] WARNING: CUDA device does not support bfloat16. Falling back to float16.")
            args.precision = 'float16'
            ptdtype = torch.float16

    print(f"[{get_timestamp()}] Using precision: {args.precision} (requested: {original_precision})")

    # Determine output filename
    if args.output is None:
        input_base = os.path.splitext(os.path.basename(args.input))[0]
        suffix = ''
        if args.tokenizer == 'bpe':
            suffix += '_bpe'
        if args.linear_attention:
            suffix += '_linear'
        args.output = f"{input_base}{suffix}.pt"

    # Ensure output goes to pt/ subdirectory
    output_dir = 'pt'
    os.makedirs(output_dir, exist_ok=True)
    args.output = os.path.join(output_dir, os.path.basename(args.output))

    # Handle resume vs fresh start
    if args.resume:
        print(f"[{get_timestamp()}] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)

        # Get saved state
        model_args = checkpoint['model_args']
        resume_iter = checkpoint['iter_num']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        saved_config = checkpoint.get('config', {})

        # Determine tokenizer meta path from checkpoint path
        # Handle regular, _iter{N}, and _final checkpoint names
        ckpt_base = args.resume.replace('.pt', '')
        if '_iter' in ckpt_base:
            # Strip _iter{N} suffix to get base name
            ckpt_base = ckpt_base.rsplit('_iter', 1)[0]
        elif ckpt_base.endswith('_final'):
            # Strip _final suffix
            ckpt_base = ckpt_base[:-6]
        meta_path = ckpt_base + '_meta.pkl'

        if not os.path.exists(meta_path):
            print(f"Error: Tokenizer metadata not found at {meta_path}")
            sys.exit(1)

        print(f"[{get_timestamp()}] Loading tokenizer from {meta_path}")
        tokenizer = load_tokenizer(meta_path)
        vocab_size = tokenizer.vocab_size

        # Override args.output to match checkpoint naming
        args.output = ckpt_base + '.pt'

        print(f"Resumed from iteration {resume_iter}, best val loss {best_val_loss:.4f}")
        print(f"Vocabulary size: {vocab_size} tokens")
    else:
        resume_iter = 0
        best_val_loss = float('inf')
        checkpoint = None

    # Load and process data
    print(f"[{get_timestamp()}] Loading data from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Data size: {len(text):,} characters")

    # Create tokenizer if not resuming
    if not args.resume:
        print(f"[{get_timestamp()}] Creating {args.tokenizer} tokenizer...")
        if args.tokenizer == 'char':
            tokenizer = CharTokenizer()
            tokenizer.train(text)
        elif args.tokenizer == 'bpe':
            tokenizer = BPETokenizer(vocab_size=args.vocab_size)
            tokenizer.train(text)
        else:
            raise ValueError(f"Unknown tokenizer: {args.tokenizer}")

        vocab_size = tokenizer.vocab_size
        print(f"Vocabulary size: {vocab_size} tokens")

        # Save tokenizer to metadata file
        meta_path = args.output.replace('.pt', '_meta.pkl')
        print(f"[{get_timestamp()}] Saving tokenizer to {meta_path}")
        tokenizer.save(meta_path)

    # Print complete vocabulary
    print_vocabulary(tokenizer)

    # Process data word-by-word, packing multiple words into blocks
    print(f"[{get_timestamp()}] Processing words into fixed-size blocks...")

    # Split text into individual words (one per line)
    words = text.strip().split('\n')
    print(f"Found {len(words):,} words")

    # Get newline token ID for padding
    padding_tokens = tokenizer.encode('\n')
    if padding_tokens:
        padding_token_id = padding_tokens[0]
    else:
        raise ValueError("Could not find newline token in vocabulary")

    # Pack multiple words into blocks
    blocks = []
    current_block = []
    words_too_long = 0

    for word in words:
        # Encode word with newline
        word_tokens = tokenizer.encode(word + '\n')

        # Skip words that exceed max_tokens_in_block
        if len(word_tokens) > args.max_tokens_in_block:
            words_too_long += 1
            continue

        # If adding this word exceeds block_size, finish current block and start new one
        if len(current_block) + len(word_tokens) > args.block_size:
            if current_block:  # Save current block if non-empty
                # Pad remaining space
                padded = current_block + [padding_token_id] * (args.block_size - len(current_block))
                blocks.append(padded)
            current_block = word_tokens
        else:
            current_block.extend(word_tokens)

    # Don't forget last block
    if current_block:
        padded = current_block + [padding_token_id] * (args.block_size - len(current_block))
        blocks.append(padded)

    if words_too_long > 0:
        print(f"Warning: Skipped {words_too_long} words exceeding max_tokens_in_block {args.max_tokens_in_block}")

    # Convert to numpy array: shape (num_blocks, block_size)
    data = np.array(blocks, dtype=np.uint32)
    n_blocks = len(data)
    print(f"Created {n_blocks:,} blocks of size {args.block_size}")

    # Split into train and validation by blocks
    val_blocks = int(n_blocks * args.val_split)
    train_data = data[:n_blocks - val_blocks]
    val_data = data[n_blocks - val_blocks:]
    train_tokens = len(train_data) * args.block_size

    print(f"Train blocks: {len(train_data):,} ({train_tokens:,} tokens)")
    print(f"Val blocks: {len(val_data):,}")

    # Spot check: show first few blocks
    print(f"\n[{get_timestamp()}] Spot check - first 3 training blocks:")
    for i in range(min(3, len(train_data))):
        block_tokens = train_data[i]
        decoded = tokenizer.decode(block_tokens.tolist())
        # Show with visible newlines
        decoded_visible = decoded.replace('\n', '\\n')
        print(f"  Block {i}: [{decoded_visible}]")
        print(f"    Token IDs: {block_tokens[:10].tolist()}... (first 10 of {len(block_tokens)})")
    print()

    # Model initialization
    print(f"[{get_timestamp()}] Initializing model...")

    # Use linear attention if specified
    use_linear = args.linear_attention
    attn_type = "linear" if use_linear else "softmax"

    # Determine weight tying based on argument
    tie_weights = not args.untie_weights

    if args.resume:
        # When resuming, use saved model_args
        print("Using model configuration from checkpoint...")
    else:
        # Create new model configuration
        model_args = dict(
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            block_size=args.block_size,
            dropout=args.dropout,
            vocab_size=vocab_size,
            use_linear_attention=use_linear,
            tie_weights=tie_weights,
        )

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    # Load model weights if resuming
    if args.resume:
        model.load_state_dict(checkpoint['model'])

    model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"Weight tying: {'enabled' if tie_weights else 'disabled'}")

    # Set up optimizer
    if args.resume and 'optimizer' in checkpoint:
        print("Loading optimizer state from checkpoint...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("Creating new optimizer...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Set up gradient scaler for mixed precision (if applicable)
    scaler = None
    if args.precision in ['float16', 'bfloat16'] and device == 'cuda':
        scaler = torch.cuda.amp.GradScaler(enabled=(args.precision == 'float16'))

    # Set up autocast context
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

    # Training loop
    print(f"\n[{get_timestamp()}] Starting training...")
    print(f"Tokenizer: {tokenizer.tokenizer_type}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Attention type: {attn_type}")
    print(f"Precision: {args.precision}")
    print(f"Max iterations: {args.max_iters}")
    print(f"Batch size: {args.batch_size}")
    print(f"Block size: {args.block_size}")
    print(f"Max tokens in block: {args.max_tokens_in_block}")
    print(f"Learning rate: {args.learning_rate:.2e}")
    print(f"Warmup iters: {args.warmup_iters}")
    print(f"Eval interval: {args.eval_interval}")
    print(f"Save interval: {args.save_interval}")
    print(f"Checkpoints will be saved to: {args.output}")
    if args.resume:
        print(f"Resuming from iteration: {resume_iter}")
    print("="*60)

    iter_num = resume_iter
    # best_val_loss already set above (from checkpoint or inf)
    running_mfu = -1.0

    # Get first batch
    X, Y = get_batch(train_data, args.batch_size, args.block_size, device)

    # Initialize timing
    t0 = time.time()
    local_iter_num = 0
    raw_model = model

    while iter_num < args.max_iters:
        # Determine and set the learning rate for this iteration
        lr = get_lr(iter_num, args.learning_rate, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluate the loss on train and val sets
        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data,
                                 args.batch_size, args.block_size,
                                 args.eval_iters, device)
            epoch = calculate_epoch(iter_num, args.batch_size, args.block_size, train_tokens)
            print(f"[{get_timestamp()}] Step {iter_num:5d} | Epoch {epoch:6.2f} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | lr {lr:.2e}")

            # Check for NaN
            if math.isnan(losses['train']) or math.isnan(losses['val']):
                print("\n" + "!"*60)
                print("WARNING: NaN detected in loss! Stopping training.")
                print("!"*60)
                break

            # Check for loss explosion
            if losses['train'] > 100 or losses['val'] > 100:
                print("WARNING: Loss explosion detected! Consider reducing learning rate.")

            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if iter_num > 0:  # Don't save initial model
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': vars(args),
                        'precision': args.precision,
                        'tokenizer_type': tokenizer.tokenizer_type,
                    }
                    torch.save(checkpoint, args.output)
                    print(f"  ✔ Saved checkpoint to {args.output} (best val loss: {best_val_loss:.4f})")

        # Save checkpoint at regular intervals
        if iter_num > 0 and iter_num % args.save_interval == 0:
            checkpoint_file = os.path.splitext(args.output)[0] + f'_iter{iter_num}.pt'
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'val_loss': losses.get('val', None) if 'losses' in locals() else None,
                'config': vars(args),
                'tokenizer_type': tokenizer.tokenizer_type,
            }
            torch.save(checkpoint, checkpoint_file)
            epoch = calculate_epoch(iter_num, args.batch_size, args.block_size, train_tokens)
            print(f"[{get_timestamp()}] Saved periodic checkpoint to {checkpoint_file} (iter {iter_num}, epoch {epoch:.2f})")

        # Sample from the model periodically
        if iter_num % 500 == 0 and iter_num > 0:
            epoch = calculate_epoch(iter_num, args.batch_size, args.block_size, train_tokens)
            print(f"\n[{get_timestamp()}] Sampling at iter {iter_num} (epoch {epoch:.2f}):")
            model.eval()

            # Get newline token ID for stopping
            newline_token_id = padding_token_id

            # Generate multiple words
            num_samples = 5
            samples = []
            for _ in range(num_samples):
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
                with torch.no_grad():
                    generated = []
                    for _ in range(50):  # Max 50 tokens per word
                        logits, _ = model(context)
                        logits = logits[:, -1, :] / 0.8  # temperature
                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        generated.append(next_token.item())
                        if next_token.item() == newline_token_id:
                            break
                        context = torch.cat([context, next_token], dim=1)
                # Decode without the trailing newline
                word_tokens = [t for t in generated if t != newline_token_id]
                if word_tokens:
                    samples.append(tokenizer.decode(word_tokens))

            print(' '.join(samples))
            model.train()
            print()

        # Forward backward update
        with ctx:
            logits, loss = model(X, Y)

        # Check for NaN in loss immediately
        if torch.isnan(loss):
            print(f"\n" + "!"*60)
            print(f"WARNING: NaN detected at iteration {iter_num}! Stopping training.")
            print("!"*60)
            break

        # Backward pass with optional gradient scaling
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        # Speed monitoring every 100 iterations
        if iter_num % 100 == 0 and iter_num > 0:
            epoch = calculate_epoch(iter_num, args.batch_size, args.block_size, train_tokens)
            print(f"[{get_timestamp()}] Speed at iter {iter_num} (epoch {epoch:.2f}): {100/dt:.2f} iters/sec")
            t0 = time.time()  # Reset timer for next 100 iterations

        # Regular loss logging
        if iter_num % 10 == 0:
            lossf = loss.item()
            if local_iter_num >= 5:  # Let the model warm up
                mfu = raw_model.estimate_mfu(args.batch_size * 1, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

        iter_num += 1
        local_iter_num += 1

        # Get next batch
        X, Y = get_batch(train_data, args.batch_size, args.block_size, device)

    # Final save
    final_epoch = calculate_epoch(args.max_iters, args.batch_size, args.block_size, train_tokens)
    print(f"\n[{get_timestamp()}] Training complete!")
    print(f"Tokenizer: {tokenizer.tokenizer_type}")
    print(f"Attention type: {attn_type}")
    print(f"Final iteration: {args.max_iters}, Final epoch: {final_epoch:.2f}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Save final checkpoint
    if iter_num > 0:
        final_checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'config': vars(args),
            'tokenizer_type': tokenizer.tokenizer_type,
        }
        final_file = os.path.splitext(args.output)[0] + '_final.pt'
        print(f"[{get_timestamp()}] Saving final model to {final_file}")
        torch.save(final_checkpoint, final_file)

    print(f"Final model saved to: {args.output}")
    print("="*60)

    # Final sample from the model
    print(f"\n[{get_timestamp()}] Final sample from trained model:")
    model.eval()

    # Generate multiple words for final sample
    newline_token_id = padding_token_id
    num_samples = 10
    samples = []
    for _ in range(num_samples):
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        with torch.no_grad():
            generated = []
            for _ in range(50):  # Max 50 tokens per word
                logits, _ = model(context)
                logits = logits[:, -1, :] / 0.8  # temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated.append(next_token.item())
                if next_token.item() == newline_token_id:
                    break
                context = torch.cat([context, next_token], dim=1)
        # Decode without the trailing newline
        word_tokens = [t for t in generated if t != newline_token_id]
        if word_tokens:
            samples.append(tokenizer.decode(word_tokens))

    print(' '.join(samples))


if __name__ == "__main__":
    main()
