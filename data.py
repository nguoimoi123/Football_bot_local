# Xử lý dữ liệu nâng cao với data augmentation
import torch
from torch.utils.data import TensorDataset, DataLoader
from underthesea import word_tokenize
from collections import Counter
import random
import json
from config import *

class Vocabulary:
    """Quản lý từ điển thông minh với OOV handling"""
    def __init__(self):
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        self.token2id = {}
        self.id2token = {}
        self.token_counts = Counter()
        self.build_initial()
    
    def build_initial(self):
        """Khởi tạo với các token đặc biệt"""
        for idx, token in enumerate(self.special_tokens):
            self.token2id[token] = idx
            self.id2token[idx] = token
    
    def build_from_conversations(self, conversations, min_freq=2):
        """Xây dựng từ điển từ conversations"""
        all_tokens = []
        
        for conversation in conversations:
            conversation = conversation.replace("_", " ")
            tokens = word_tokenize(conversation.lower(), format="text").split()
            all_tokens.extend(tokens)
            self.token_counts.update(tokens)
        
        # Thêm tokens đạt ngưỡng tần suất
        idx = len(self.special_tokens)
        for token, count in self.token_counts.items():
            if count >= min_freq and token not in self.token2id:
                self.token2id[token] = idx
                self.id2token[idx] = token
                idx += 1
        
        print(f"✅ Từ điển được xây dựng với {len(self.token2id)} tokens")
    
    def encode(self, text, add_special_tokens=True):
        """Encode text thành token IDs"""
        tokens = word_tokenize(text.lower(), format="text").split()
        
        if add_special_tokens:
            tokens = ["<SOS>"] + tokens + ["<EOS>"]
        
        token_ids = []
        for token in tokens:
            if token in self.token2id:
                token_ids.append(self.token2id[token])
            else:
                token_ids.append(self.token2id["<UNK>"])
        
        return token_ids
    
    def decode(self, token_ids):
        """Decode token IDs thành text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id2token:
                token = self.id2token[token_id]
                if token not in self.special_tokens:
                    tokens.append(token)
            else:
                tokens.append("<UNK>")
        
        return " ".join(tokens)
    
    def __len__(self):
        return len(self.token2id)

def augment_vietnamese_text(text):
    """Data augmentation cho tiếng Việt"""
    # Synonym replacement đơn giản
    synonyms = {
        "bóng đá": "bóng",
        "cầu thủ": "cầu thủ",
        "đội bóng": "đội",
        "ghi bàn": "ghi bàn thắng",
        "trận đấu": "trận",
        "thắng": "chiến thắng",
        "thua": "thất bại"
    }
    
    words = text.split()
    augmented = []
    
    for word in words:
        if word in synonyms and random.random() < 0.3:
            augmented.append(synonyms[word])
        else:
            augmented.append(word)
    
    return " ".join(augmented)

def load_and_preprocess_data(filepath="data.txt", augment=False):
    """Tải và tiền xử lý dữ liệu nâng cao"""
    conversations = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and '<EOS>' in line:
                conversations.append(line)
                
                # Data augmentation
                if augment and random.random() < 0.5:
                    augmented = augment_vietnamese_text(line)
                    conversations.append(augmented)
    
    print(f"📊 Tải {len(conversations)} mẫu dữ liệu")
    return conversations

def create_enhanced_dataset(conversations, vocab):
    """Tạo dataset với padding và masking thông minh"""
    inputs, targets = [], []
    
    for conversation in conversations:
        if '<EOS>' in conversation:
            parts = conversation.split('<EOS>')
            if len(parts) >= 2:
                question, answer = parts[0].strip(), parts[1].strip()
                
                # Encode
                question_ids = vocab.encode(question, add_special_tokens=True)
                answer_ids = vocab.encode(answer, add_special_tokens=True)
                
                # Padding thông minh
                question_ids = _smart_padding(question_ids, MAX_LEN, vocab.token2id["<PAD>"])
                answer_ids = _smart_padding(answer_ids, MAX_LEN, vocab.token2id["<PAD>"])
                
                inputs.append(question_ids)
                targets.append(answer_ids)
    
    return torch.tensor(inputs), torch.tensor(targets)

def _smart_padding(token_ids, max_len, pad_token_id):
    """Padding thông minh - giữ nguyên thông tin quan trọng"""
    if len(token_ids) >= max_len:
        # Giữ <SOS> và <EOS>, cắt phần giữa
        return token_ids[:1] + token_ids[1:-1][:max_len-2] + token_ids[-1:]
    else:
        # Padding thông thường
        return token_ids + [pad_token_id] * (max_len - len(token_ids))

def get_enhanced_dataloader(filepath="data.txt", augment=True):
    """Tạo DataLoader nâng cao"""
    # Tải và tiền xử lý dữ liệu
    conversations = load_and_preprocess_data(filepath, augment)
    
    # Xây dựng từ điển
    vocab = Vocabulary()
    vocab.build_from_conversations(conversations)
    
    # Tạo dataset
    inputs, targets = create_enhanced_dataset(conversations, vocab)
    
    # Tạo DataLoader với các kỹ thuật tối ưu
    dataset = TensorDataset(inputs, targets)
    dataset.vocab = vocab  # Lưu vocab vào dataset
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,  # Tốt cho GPU
        persistent_workers=True,
        prefetch_factor=2  # Tải trước dữ liệu
    )
    
    return dataloader

def save_vocabulary(vocab, filepath="vocabulary.json"):
    """Lưu từ điển để sử dụng sau"""
    vocab_data = {
        "token2id": vocab.token2id,
        "id2token": {int(k): v for k, v in vocab.id2token.items()},
        "token_counts": dict(vocab.token_counts)
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)

def load_vocabulary(filepath="vocabulary.json"):
    """Tải từ điển đã lưu"""
    vocab = Vocabulary()
    
    with open(filepath, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
        
    vocab.token2id = vocab_data["token2id"]
    vocab.id2token = {int(k): v for k, v in vocab_data["id2token"].items()}
    vocab.token_counts = Counter(vocab_data["token_counts"])
    
    return vocab