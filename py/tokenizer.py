"""
Unified tokenizer interface for character-level and BPE tokenization.

Provides CharTokenizer (existing character-level approach) and BPETokenizer
(using tiktoken for 16K vocabulary) with identical interfaces.
"""

import pickle
from abc import ABC, abstractmethod
from typing import List, Dict
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


class Tokenizer(ABC):
    """Abstract base class for tokenizers"""
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Convert text to list of token IDs"""
        pass
    
    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Convert list of token IDs back to text"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save tokenizer state to file"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load tokenizer state from file"""
        pass
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return vocabulary size"""
        pass
    
    @property
    @abstractmethod
    def tokenizer_type(self) -> str:
        """Return tokenizer type identifier"""
        pass


class CharTokenizer(Tokenizer):
    """
    Character-level tokenizer.
    Maps each unique character to an integer ID.
    """
    
    def __init__(self):
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self._vocab_size = 0
    
    def train(self, text: str) -> None:
        """Build vocabulary from text"""
        chars = sorted(list(set(text)))
        self._vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        print(f"CharTokenizer: Built vocabulary of {self._vocab_size} characters")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs, using 0 for unknown characters"""
        return [self.stoi.get(c, 0) for c in text]
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text, using '?' for unknown IDs"""
        return ''.join([self.itos.get(i, '?') for i in tokens])
    
    def save(self, path: str) -> None:
        """Save vocabulary mappings"""
        data = {
            'tokenizer_type': 'char',
            'stoi': self.stoi,
            'itos': self.itos,
            'vocab_size': self._vocab_size
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str) -> None:
        """Load vocabulary mappings"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        assert data['tokenizer_type'] == 'char', "Not a character tokenizer file"
        self.stoi = data['stoi']
        self.itos = data['itos']
        self._vocab_size = data['vocab_size']
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    @property
    def tokenizer_type(self) -> str:
        return 'char'


class BPETokenizer(Tokenizer):
    """
    BPE tokenizer using HuggingFace tokenizers library.
    Trains a custom vocabulary of specified size on provided corpus.
    """
    
    def __init__(self, vocab_size: int = 16384):
        self.target_vocab_size = vocab_size
        self.tokenizer: HFTokenizer = None
        self._vocab_size = 0
    
    def train(self, text: str) -> None:
        """
        Train BPE vocabulary on text.
        Uses HuggingFace tokenizers library with ByteLevel pre-tokenizer
        which preserves all characters including spaces.
        """
        print(f"BPETokenizer: Training vocabulary of {self.target_vocab_size} tokens...")
        print(f"Training on {len(text):,} characters...")
        
        # Initialize BPE tokenizer
        self.tokenizer = HFTokenizer(BPE(unk_token="[UNK]"))
        
        # Use ByteLevel pre-tokenizer (like GPT-2) which preserves all bytes
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        
        # Set up BPE trainer
        trainer = BpeTrainer(
            vocab_size=self.target_vocab_size,
            special_tokens=["[UNK]"],
            show_progress=True
        )
        
        # Write text to temporary file for training
        # (HF tokenizers expects file paths)
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write(text)
            temp_file = f.name
        
        # Train the tokenizer
        self.tokenizer.train([temp_file], trainer)
        
        # Clean up temp file
        import os
        os.remove(temp_file)
        
        # Set ByteLevel decoder to properly decode
        self.tokenizer.decoder = ByteLevelDecoder()
        
        self._vocab_size = self.tokenizer.get_vocab_size()
        print(f"BPETokenizer: Trained vocabulary with {self._vocab_size} tokens")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        return self.tokenizer.encode(text).ids
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        return self.tokenizer.decode(tokens)
    
    def save(self, path: str) -> None:
        """Save trained BPE tokenizer"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained")
        
        # Save the HF tokenizer to a JSON file
        json_path = path.replace('.pkl', '.json')
        self.tokenizer.save(json_path)
        
        # Also save metadata in pickle format for consistency
        data = {
            'tokenizer_type': 'bpe',
            'vocab_size': self._vocab_size,
            'target_vocab_size': self.target_vocab_size,
            'json_path': json_path
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved BPE tokenizer to {path} and {json_path}")
    
    def load(self, path: str) -> None:
        """Load trained BPE tokenizer"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        assert data['tokenizer_type'] == 'bpe', "Not a BPE tokenizer file"
        
        self.target_vocab_size = data['target_vocab_size']
        self._vocab_size = data['vocab_size']
        
        # Compute json_path from pickle path (don't use stored path which may be stale)
        json_path = path.replace('.pkl', '.json')
        self.tokenizer = HFTokenizer.from_file(json_path)
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    @property
    def tokenizer_type(self) -> str:
        return 'bpe'


def load_tokenizer(path: str) -> Tokenizer:
    """
    Load a tokenizer from file.
    Automatically detects type and returns appropriate tokenizer instance.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    tokenizer_type = data['tokenizer_type']
    
    if tokenizer_type == 'char':
        tokenizer = CharTokenizer()
    elif tokenizer_type == 'bpe':
        tokenizer = BPETokenizer(vocab_size=data['target_vocab_size'])
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
    
    # Load the full tokenizer state
    with open(path, 'rb') as f:
        if tokenizer_type == 'char':
            tokenizer.load(path)
        elif tokenizer_type == 'bpe':
            tokenizer.load(path)
    
    return tokenizer


# Convenience function for backward compatibility
def create_char_tokenizer(text: str) -> CharTokenizer:
    """Create and train a character tokenizer on text"""
    tokenizer = CharTokenizer()
    tokenizer.train(text)
    return tokenizer


def create_bpe_tokenizer(text: str, vocab_size: int = 16384) -> BPETokenizer:
    """Create and train a BPE tokenizer on text"""
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(text)
    return tokenizer
