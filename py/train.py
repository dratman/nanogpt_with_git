#!/usr/bin/env python3
"""
Training script supporting both sentence-per-block and word-per-block training.

Modes:
- sentence: One sentence per block with padding (for prose corpora)
- word: One word per block with padding (for word lists)

Supports three attention mechanisms:
- softmax (default): Standard transformer attention
- linear: O(N) linear attention
- autocorrelation: FFT-based autocorrelation attention (Autoformer-inspired)

Usage:
  # Sentence mode (default) - for prose corpora
  python train.py --input corpus.txt --mode sentence --block_size 128 --batch_size 64

  # Word mode - for word lists (one word per line)
  python train.py --input wordlist.txt --mode word --block_size 18 --batch_size 64

  # With autocorrelation attention
  python train.py --input corpus.txt --autocorrelation_attention
"""

import argparse
import os
import sys
import time
import math
import pickle
import re
import random
import numpy as np
import torch
from datetime import datetime
from contextlib import nullcontext
from model import GPT, GPTConfig
from tokenizer import CharTokenizer, BPETokenizer, load_tokenizer


# Padding token ID - always at vocab_size
PADDING_TOKEN = None  # Will be set after tokenizer is initialized


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train GPT model with one-sentence-per-block or one-word-per-block',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--input', type=str, required=True,
                       help='Input text file (prose corpus for sentence mode, word list for word mode)')

    # Mode selection
    parser.add_argument('--mode', type=str, default='sentence',
                       choices=['sentence', 'word'],
                       help='Training mode: sentence (prose) or word (word list)')

    # Optional arguments with defaults
    parser.add_argument('--output', type=str, default=None,
                       help='Output model file name (default: based on input name)')
    parser.add_argument('--checkpoints_to', type=str, default='pt',
                       help='Directory for checkpoint files')
    parser.add_argument('--n_layer', type=int, default=1,
                       help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=256,
                       help='Embedding dimension')
    parser.add_argument('--block_size', type=int, default=128,
                       help='Maximum sequence length in tokens')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--max_iters', type=int, default=30000,
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
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Loss logging interval')
    parser.add_argument('--sample_interval', type=int, default=500,
                       help='How often to sample from the model')
    parser.add_argument('--sample_max_tokens', type=int, default=128,
                       help='Maximum tokens to generate when sampling')

    # Attention mechanism options (mutually exclusive)
    parser.add_argument('--linear_attention', action='store_true',
                       help='Use linear attention instead of softmax attention')
    parser.add_argument('--autocorrelation_attention', action='store_true',
                       help='Use autocorrelation attention (FFT-based, Autoformer-inspired)')

    # Weight tying option
    parser.add_argument('--untie_weights', action='store_true',
                       help='Do not tie input embeddings to output projection')

    # GELU option
    parser.add_argument('--no_gelu', action='store_true',
                       help='Disable GELU nonlinearity in MLP (makes it purely linear)')

    # Precision option
    parser.add_argument('--precision', type=str, default='float32',
                       choices=['float16', 'float32', 'float64', 'bfloat16'],
                       help='Floating-point precision')

    # Tokenizer options
    parser.add_argument('--tokenizer', type=str, default='bpe',
                       choices=['char', 'bpe'],
                       help='Tokenization scheme: bpe (default) or char')
    parser.add_argument('--vocab_size', type=int, default=8192,
                       help='Vocabulary size for BPE tokenizer (ignored for char)')

    # Resume option
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')

    # Sentence mode specific options
    parser.add_argument('--min_sentence_tokens', type=int, default=5,
                       help='Minimum sentence length in tokens (skip shorter) [sentence mode only]')
    parser.add_argument('--max_sentences', type=int, default=None,
                       help='Maximum number of sentences to use (None = all) [sentence mode only]')

    # Debug output
    parser.add_argument('--debug', action='store_true',
                       help='Enable verbose debug output')

    args = parser.parse_args()

    # Validate mutually exclusive attention options
    if args.linear_attention and args.autocorrelation_attention:
        parser.error("Cannot use both --linear_attention and --autocorrelation_attention")

    return args


def get_timestamp():
    """Get current timestamp in readable format"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_elapsed(start_time):
    """Format elapsed time since start_time as 'Xd Xh Xm'"""
    elapsed = time.time() - start_time
    days = int(elapsed // 86400)
    hours = int((elapsed % 86400) // 3600)
    minutes = int((elapsed % 3600) // 60)
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


def calculate_epoch(iter_num, batch_size, block_size, train_tokens):
    """Calculate approximate epoch from iteration number"""
    tokens_per_iter = batch_size * block_size
    tokens_processed = iter_num * tokens_per_iter
    epoch = tokens_processed / train_tokens
    return epoch


# =============================================================================
# Sentence Mode Functions
# =============================================================================

def split_into_sentences(text):
    """
    Split text into sentences on sentence-ending punctuation.
    Handles common abbreviations and edge cases.

    Returns list of sentences (strings), each ending with its punctuation.
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Split on sentence boundaries: .!? followed by space and capital letter or end
    # This regex captures the punctuation with the sentence
    # Pattern: split after .!? when followed by space+capital or end of string

    # Simple approach: split on .!? followed by whitespace
    # Then filter out empty strings and very short fragments

    # Use a regex that keeps the delimiter with the sentence
    pattern = r'(?<=[.!?])\s+'
    raw_sentences = re.split(pattern, text)

    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if len(s) > 0:
            sentences.append(s)

    return sentences


def prepare_sentence_data(text, tokenizer, block_size, padding_token,
                          min_tokens=5, max_sentences=None, debug=False):
    """
    Prepare data for one-sentence-per-block training.

    Each sentence is tokenized with a leading newline (BOS) and trailing newline (EOS).
    Input: [BOS, tok1, tok2, ..., tokN]
    Target: [tok1, tok2, ..., tokN, EOS]

    Returns:
        sentence_blocks: list of (input_block, target_block) tuples
        Each block is padded to block_size with padding_token
    """
    sentences = split_into_sentences(text)

    if max_sentences is not None:
        sentences = sentences[:max_sentences]

    print(f"[{get_timestamp()}] Found {len(sentences)} sentences in corpus")

    sentence_blocks = []
    skipped_short = 0
    skipped_long = 0

    # Get newline token for BOS/EOS
    newline_tokens = tokenizer.encode('\n')
    if len(newline_tokens) != 1:
        print(f"Warning: newline encodes to {len(newline_tokens)} tokens, expected 1")
    newline_token = newline_tokens[0]

    print("Tokenizing sentences...")
    for i, sentence in enumerate(sentences):
        # Tokenize sentence
        sentence_tokens = tokenizer.encode(sentence)

        # Skip very short sentences
        if len(sentence_tokens) < min_tokens:
            skipped_short += 1
            continue

        # Add BOS (newline) at start and EOS (newline) at end
        # Full sequence: [newline, sentence_tokens..., newline]
        full_tokens = [newline_token] + sentence_tokens + [newline_token]

        # Skip sentences that are too long (need room for input/target split)
        if len(full_tokens) > block_size:
            skipped_long += 1
            continue

        # Create input and target sequences
        # Input: all but last token [BOS, tok1, tok2, ..., tokN]
        # Target: all but first token [tok1, tok2, ..., tokN, EOS]
        input_tokens = full_tokens[:-1]
        target_tokens = full_tokens[1:]

        # Pad to block_size
        input_padded = input_tokens + [padding_token] * (block_size - len(input_tokens))
        target_padded = target_tokens + [padding_token] * (block_size - len(target_tokens))

        sentence_blocks.append((input_padded, target_padded))

        if debug and i < 3:
            print(f"  Example {i}: '{sentence[:50]}...'")
            print(f"    Tokens: {len(sentence_tokens)}, Full: {len(full_tokens)}")

    print(f"[{get_timestamp()}] Created {len(sentence_blocks)} sentence blocks")
    print(f"[{get_timestamp()}] Skipped {skipped_short} sentences (too short, <{min_tokens} tokens)")
    print(f"[{get_timestamp()}] Skipped {skipped_long} sentences (too long, >{block_size} tokens)")

    # Report token length statistics
    if sentence_blocks:
        lengths = [len([t for t in block[0] if t != padding_token]) for block in sentence_blocks]
        print(f"[{get_timestamp()}] Token lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")

    return sentence_blocks


# =============================================================================
# Word Mode Functions
# =============================================================================

def prepare_word_data(text, tokenizer, block_size, padding_token, debug=False):
    """
    Prepare data for one-word-per-block training.

    Returns:
        word_blocks: list of (input_block, target_block) tuples
        Each block is padded to block_size with padding_token
        Words longer than block_size-1 are skipped
    """
    words = text.strip().split('\n')
    word_blocks = []
    skipped_count = 0

    print(f"[{get_timestamp()}] Preparing word blocks from {len(words)} words...")

    for i, word in enumerate(words):
        # Encode word with leading and trailing newline
        # This teaches: given \n, predict first letter; given last letter, predict \n
        word_with_newlines = '\n' + word + '\n'
        tokens = tokenizer.encode(word_with_newlines)

        # Skip words that are too long (need room for at least one target token)
        if len(tokens) > block_size - 1:
            skipped_count += 1
            continue

        # Create input and target sequences
        # Input: [\n, w, o, r, d] (all but final \n)
        # Target: [w, o, r, d, \n] (all but leading \n)
        input_tokens = tokens[:-1]  # All but last
        target_tokens = tokens[1:]   # All but first

        # Pad to block_size
        input_padded = input_tokens + [padding_token] * (block_size - len(input_tokens))
        target_padded = target_tokens + [padding_token] * (block_size - len(target_tokens))

        word_blocks.append((input_padded, target_padded))

        if debug and i < 3:
            print(f"  Example {i}: '{word}'")
            print(f"    Tokens: {len(tokens)}")

    print(f"[{get_timestamp()}] Created {len(word_blocks)} word blocks")
    print(f"[{get_timestamp()}] Skipped {skipped_count} words (too long for block_size={block_size})")

    return word_blocks


# =============================================================================
# Common Functions
# =============================================================================

def get_batch(blocks, batch_size, device='cpu'):
    """
    Generate a batch of blocks for training.
    Each batch contains batch_size randomly selected blocks.
    Works for both sentence and word blocks.
    """
    indices = torch.randint(len(blocks), (batch_size,))

    x_list = []
    y_list = []

    for idx in indices:
        input_block, target_block = blocks[idx]
        x_list.append(input_block)
        y_list.append(target_block)

    x = torch.tensor(x_list, dtype=torch.long, device=device)
    y = torch.tensor(y_list, dtype=torch.long, device=device)

    return x, y


@torch.no_grad()
def estimate_loss(model, train_blocks, val_blocks, batch_size, eval_iters, device):
    """Estimate loss on train and validation sets"""
    out = {}
    model.eval()
    for split, blocks in [('train', train_blocks), ('val', val_blocks)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(blocks, batch_size, device)
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


# =============================================================================
# Sampling Functions
# =============================================================================

@torch.no_grad()
def sample_sentences(model, tokenizer, device, num_sentences=5, max_tokens=100,
                     temperature=0.8, top_k=40, padding_token=None, debug=False):
    """
    Generate sample sentences from the model.
    Start with newline (BOS), generate until newline (EOS) or max_tokens.
    """
    model.eval()

    newline_token = tokenizer.encode('\n')[0]
    sentences = []

    for i in range(num_sentences):
        # Start with BOS (newline)
        context = torch.tensor([[newline_token]], dtype=torch.long, device=device)

        generated = model.generate(context, max_new_tokens=max_tokens,
                                   temperature=temperature, top_k=top_k)[0].tolist()

        if debug:
            print(f"DEBUG sentence {i}: Raw generated tokens: {generated[:15]}...")

        # Skip BOS and find EOS (newline)
        generated = generated[1:]  # Skip initial newline
        if newline_token in generated:
            eos_idx = generated.index(newline_token)
            generated = generated[:eos_idx]
            if debug:
                print(f"DEBUG sentence {i}: Found EOS at index {eos_idx}")

        # Filter out padding tokens if any
        if padding_token is not None:
            generated = [t for t in generated if t != padding_token]

        sentence = tokenizer.decode(generated)
        sentences.append(sentence)

        if debug:
            print(f"DEBUG sentence {i}: Decoded: '{sentence[:50]}...'")

    model.train()
    return sentences


@torch.no_grad()
def sample_words(model, tokenizer, device, num_words=20, max_tokens=20,
                 temperature=0.8, top_k=40, padding_token=None, debug=False):
    """
    Generate sample words from the model.
    Start with newline (BOS), generate until newline (EOS) or max_tokens.
    """
    model.eval()

    newline_token = tokenizer.encode('\n')[0]
    words = []

    if debug:
        print(f"DEBUG: Newline token ID: {newline_token}")

    for i in range(num_words):
        # Start with BOS (newline)
        context = torch.tensor([[newline_token]], dtype=torch.long, device=device)

        generated = model.generate(context, max_new_tokens=max_tokens,
                                   temperature=temperature, top_k=top_k)[0].tolist()

        if debug:
            print(f"DEBUG word {i}: Raw generated tokens: {generated[:10]}")

        # Skip BOS and find EOS (newline)
        generated = generated[1:]  # Skip initial newline
        if newline_token in generated:
            newline_idx = generated.index(newline_token)
            generated = generated[:newline_idx + 1]  # Include the newline for word mode
            if debug:
                print(f"DEBUG word {i}: Found newline at index {newline_idx}")

        # Filter out padding tokens
        if padding_token is not None:
            generated = [t for t in generated if t != padding_token]

        if debug:
            print(f"DEBUG word {i}: Filtered tokens: {generated}")

        word = tokenizer.decode(generated)
        words.append(word)

        if debug:
            print(f"DEBUG word {i}: Decoded word: '{word}'")

    model.train()
    return words


def main():
    global PADDING_TOKEN

    args = parse_args()

    # Validate tokenizer arguments
    if not args.resume and args.tokenizer == 'char' and args.vocab_size != 8192:
        print("Warning: --vocab_size is ignored with --tokenizer char")

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

    # Check device compatibility
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
    print(f"[{get_timestamp()}] Training mode: {args.mode}")

    # Determine attention type for naming
    if args.autocorrelation_attention:
        attn_suffix = '_autocorr'
        attn_type = 'autocorrelation'
    elif args.linear_attention:
        attn_suffix = '_linear'
        attn_type = 'linear'
    else:
        attn_suffix = ''
        attn_type = 'softmax'

    # Determine output filename
    if args.output is None:
        input_base = os.path.splitext(os.path.basename(args.input))[0]
        suffix = f'_{args.mode}'  # Mark training mode
        if args.tokenizer == 'bpe':
            suffix += '_bpe'
        suffix += attn_suffix
        if args.untie_weights:
            suffix += '_untied'
        args.output = f'{args.checkpoints_to}/{input_base}{suffix}.pt'

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    # Read input text
    print(f"[{get_timestamp()}] Reading input from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"[{get_timestamp()}] Read {len(text):,} characters")

    # Check if resuming
    resume_iter = 0
    best_val_loss = float('inf')

    if args.resume:
        print(f"[{get_timestamp()}] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        resume_iter = checkpoint.get('iter_num', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        # Load tokenizer from checkpoint metadata
        meta_path = args.resume.replace('.pt', '_meta.pkl')
        if not os.path.exists(meta_path):
            print(f"Error: Metadata file '{meta_path}' not found")
            sys.exit(1)
        print(f"[{get_timestamp()}] Loading tokenizer from {meta_path}")
        tokenizer = load_tokenizer(meta_path)
        vocab_size = tokenizer.vocab_size
    else:
        # Create tokenizer
        print(f"[{get_timestamp()}] Training {args.tokenizer} tokenizer...")
        if args.tokenizer == 'char':
            tokenizer = CharTokenizer()
            tokenizer.train(text)
            vocab_size = tokenizer.vocab_size
        else:  # bpe
            tokenizer = BPETokenizer(vocab_size=args.vocab_size)
            tokenizer.train(text)
            vocab_size = tokenizer.vocab_size

    # Set padding token
    PADDING_TOKEN = vocab_size
    print(f"[{get_timestamp()}] Padding token ID: {PADDING_TOKEN}")

    # Prepare data blocks based on mode
    if args.mode == 'sentence':
        data_blocks = prepare_sentence_data(
            text, tokenizer, args.block_size, PADDING_TOKEN,
            min_tokens=args.min_sentence_tokens,
            max_sentences=args.max_sentences,
            debug=args.debug
        )
    else:  # word mode
        data_blocks = prepare_word_data(
            text, tokenizer, args.block_size, PADDING_TOKEN,
            debug=args.debug
        )

    if len(data_blocks) == 0:
        print("Error: No valid data blocks found. Check block_size and input data.")
        sys.exit(1)

    # Split into train and validation
    n = len(data_blocks)
    n_val = int(n * args.val_split)
    n_train = n - n_val

    # Shuffle data blocks
    random.shuffle(data_blocks)

    train_blocks = data_blocks[:n_train]
    val_blocks = data_blocks[n_train:]

    # Calculate total training tokens for epoch calculation
    train_tokens = len(train_blocks) * args.block_size

    print(f"[{get_timestamp()}] Training blocks: {len(train_blocks):,} ({train_tokens:,} tokens)")
    print(f"[{get_timestamp()}] Validation blocks: {len(val_blocks)}")

    # Save metadata
    if not args.resume:
        meta_path = os.path.splitext(args.output)[0] + '_meta.pkl'
        print(f"[{get_timestamp()}] Saving tokenizer to {meta_path}")
        tokenizer.save(meta_path)

    # Initialize or load model
    if args.resume:
        print(f"[{get_timestamp()}] Loading model from checkpoint...")
        model_args = checkpoint['model_args']
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
    else:
        # Model needs vocab_size + 1 to accommodate padding token
        model_vocab_size = vocab_size + 1

        print(f"[{get_timestamp()}] Initializing new model...")
        print(f"  Vocabulary size: {vocab_size} (+ 1 padding token = {model_vocab_size})")

        model_args = dict(
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            block_size=args.block_size,
            bias=True,
            vocab_size=model_vocab_size,
            dropout=args.dropout,
            use_linear_attention=args.linear_attention,
            use_autocorrelation_attention=args.autocorrelation_attention,
            tie_weights=not args.untie_weights,
            no_gelu=args.no_gelu,
        )

        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        model.to(device)

    # Modify model to use padding token in loss calculation
    def forward_with_padding(self, idx, targets=None):
        """Modified forward that ignores padding tokens in loss"""
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Forward the GPT model itself
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=PADDING_TOKEN
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    # Replace model's forward method
    import types
    model.forward = types.MethodType(forward_with_padding, model)

    # Set up optimizer
    if args.resume:
        optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=args.learning_rate,
                                              betas=(0.9, 0.95), device_type=device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"[{get_timestamp()}] Loaded optimizer state from checkpoint")
    else:
        optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=args.learning_rate,
                                              betas=(0.9, 0.95), device_type=device)

    # Set up mixed precision training
    ctx = nullcontext()
    scaler = None
    if args.precision == 'float16' and device == 'cuda':
        ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16)
        scaler = torch.amp.GradScaler('cuda')
        print(f"[{get_timestamp()}] Using automatic mixed precision (AMP)")

    # Training loop
    print(f"\n[{get_timestamp()}] Starting training...")
    print(f"Mode: {args.mode}")
    print(f"Tokenizer: {tokenizer.tokenizer_type}")
    print(f"Vocabulary size: {vocab_size} (+ 1 padding = {vocab_size + 1})")
    print(f"Padding token ID: {PADDING_TOKEN}")
    print(f"Attention type: {attn_type}")
    print(f"Precision: {args.precision}")
    print(f"Max iterations: {args.max_iters}")
    print(f"Batch size: {args.batch_size}")
    print(f"Block size: {args.block_size}")
    print(f"Learning rate: {args.learning_rate:.2e}")
    print(f"Warmup iters: {args.warmup_iters}")
    print(f"Eval interval: {args.eval_interval}")
    print(f"Save interval: {args.save_interval}")
    print(f"Log interval: {args.log_interval}")
    print(f"Sample interval: {args.sample_interval}")
    print(f"Checkpoints will be saved to: {args.output}")
    if args.resume:
        print(f"Resuming from iteration: {resume_iter}")
    print("=" * 60)

    iter_num = resume_iter
    running_mfu = -1.0

    # Get first batch
    X, Y = get_batch(train_blocks, args.batch_size, device)

    # Initialize timing
    t0 = time.time()
    speed_t0 = time.time()
    training_start_time = time.time()  # For elapsed time display
    local_iter_num = 0
    raw_model = model

    while iter_num < args.max_iters:
        # Set learning rate
        lr = get_lr(iter_num, args.learning_rate, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluate
        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(model, train_blocks, val_blocks,
                                 args.batch_size, args.eval_iters, device)
            epoch = calculate_epoch(iter_num, args.batch_size, args.block_size, train_tokens)
            elapsed = format_elapsed(training_start_time)
            print(f"[{get_timestamp()}] [{elapsed}] Step {iter_num:5d} | Epoch {epoch:6.2f} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | lr {lr:.2e}")

            # Check for NaN
            if math.isnan(losses['train']) or math.isnan(losses['val']):
                print("\n" + "!" * 60)
                print("WARNING: NaN detected in loss! Stopping training.")
                print("!" * 60)
                break

            # Check for loss explosion
            if losses['train'] > 100 or losses['val'] > 100:
                print("WARNING: Loss explosion detected! Consider reducing learning rate.")

            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': vars(args),
                        'precision': args.precision,
                        'tokenizer_type': tokenizer.tokenizer_type,
                        'padding_token': PADDING_TOKEN,
                        'mode': args.mode,
                    }
                    torch.save(checkpoint, args.output)
                    print(f"  Saved checkpoint to {args.output} (best val loss: {best_val_loss:.4f})")

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
                'padding_token': PADDING_TOKEN,
                'mode': args.mode,
            }
            torch.save(checkpoint, checkpoint_file)
            epoch = calculate_epoch(iter_num, args.batch_size, args.block_size, train_tokens)
            elapsed = format_elapsed(training_start_time)
            print(f"[{get_timestamp()}] [{elapsed}] Saved periodic checkpoint to {checkpoint_file} (iter {iter_num}, epoch {epoch:.2f})")

        # Sample from the model periodically
        if iter_num % args.sample_interval == 0 and iter_num > 0:
            epoch = calculate_epoch(iter_num, args.batch_size, args.block_size, train_tokens)
            elapsed = format_elapsed(training_start_time)
            print(f"\n[{get_timestamp()}] [{elapsed}] Sampling at iter {iter_num} (epoch {epoch:.2f}):")
            if args.mode == 'sentence':
                samples = sample_sentences(model, tokenizer, device,
                                          num_sentences=3, max_tokens=args.sample_max_tokens,
                                          temperature=0.8, top_k=40,
                                          padding_token=PADDING_TOKEN,
                                          debug=args.debug)
                for i, sent in enumerate(samples):
                    print(f"  {i+1}. {sent}")
            else:  # word mode
                words = sample_words(model, tokenizer, device,
                                    num_words=20, max_tokens=20,
                                    temperature=0.8, top_k=40,
                                    padding_token=PADDING_TOKEN,
                                    debug=args.debug)
                sample = ''.join(words)
                print(f"\nSample (first 200 chars): {sample[:200]}")
            print()

        # Forward backward update
        with ctx:
            logits, loss = model(X, Y)

        # Check for NaN
        if torch.isnan(loss):
            print(f"\n" + "!" * 60)
            print(f"WARNING: NaN detected at iteration {iter_num}! Stopping training.")
            print("!" * 60)
            break

        # Backward pass
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

        # Timing
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        # Speed monitoring
        if iter_num % 100 == 0 and iter_num > 0:
            speed_dt = time.time() - speed_t0
            epoch = calculate_epoch(iter_num, args.batch_size, args.block_size, train_tokens)
            elapsed = format_elapsed(training_start_time)
            print(f"[{get_timestamp()}] [{elapsed}] Speed at iter {iter_num} (epoch {epoch:.2f}): {100/speed_dt:.2f} iters/sec")
            speed_t0 = time.time()

        # Loss logging
        if iter_num % args.log_interval == 0:
            lossf = loss.item()
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(args.batch_size * 1, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

        iter_num += 1
        local_iter_num += 1

        # Get next batch
        X, Y = get_batch(train_blocks, args.batch_size, device)

    # Final save
    final_epoch = calculate_epoch(iter_num, args.batch_size, args.block_size, train_tokens)
    elapsed = format_elapsed(training_start_time)
    print(f"\n[{get_timestamp()}] [{elapsed}] Training complete!")
    print(f"Mode: {args.mode}")
    print(f"Tokenizer: {tokenizer.tokenizer_type}")
    print(f"Attention type: {attn_type}")
    print(f"Final iteration: {iter_num}, Final epoch: {final_epoch:.2f}")
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
            'padding_token': PADDING_TOKEN,
            'mode': args.mode,
        }
        final_file = os.path.splitext(args.output)[0] + '_final.pt'
        print(f"[{get_timestamp()}] Saving final model to {final_file}")
        torch.save(final_checkpoint, final_file)

    print(f"Final model saved to: {args.output}")
    print("=" * 60)

    # Final sample
    print(f"\n[{get_timestamp()}] Final sample from trained model:")
    if args.mode == 'sentence':
        samples = sample_sentences(model, tokenizer, device,
                                  num_sentences=5, max_tokens=args.sample_max_tokens,
                                  temperature=0.8, top_k=40,
                                  padding_token=PADDING_TOKEN,
                                  debug=args.debug)
        for i, sent in enumerate(samples):
            print(f"  {i+1}. {sent}")
    else:  # word mode
        words = sample_words(model, tokenizer, device,
                            num_words=20, max_tokens=20,
                            temperature=0.8, top_k=40,
                            padding_token=PADDING_TOKEN,
                            debug=args.debug)
        sample = ''.join(words)
        print(sample)


if __name__ == "__main__":
    main()
