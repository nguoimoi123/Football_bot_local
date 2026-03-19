# Training nâng cao với các kỹ thuật tối ưu - SIMPLIFIED VERSION
import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor
)
import torch.nn.functional as F
from config import *
from data import get_enhanced_dataloader, save_vocabulary
from model import FootballChatbot  # Sử dụng model đơn giản hơn

class SimpleTrainer:
    def __init__(self):
        self.dataloader = None
        self.model = None
        self.vocab = None
    
    def setup_training(self):
        """Thiết lập training environment - ĐƠN GIẢN HÓA"""
        # Thiết lập precision
        torch.set_float32_matmul_precision('medium')
        
        # Tải dữ liệu
        print("📥 Đang tải và xử lý dữ liệu...")
        self.dataloader = get_enhanced_dataloader(augment=False)  # Tắt augmentation để ổn định
        self.vocab = self.dataloader.dataset.vocab
        
        # Kiểm tra dữ liệu
        if len(self.dataloader) == 0:
            raise ValueError("Không có dữ liệu để training!")
        
        # Khởi tạo mô hình ĐƠN GIẢN
        print("🤖 Đang khởi tạo mô hình...")
        self.model = FootballChatbot(
            num_tokens=len(self.vocab),
            d_model=D_MODEL,
            max_len=MAX_LEN
        )
        
        # Lưu vocabulary
        save_vocabulary(self.vocab)
        
        print(f"✅ Mô hình được khởi tạo với {len(self.vocab)} tokens")
        print(f"✅ DataLoader có {len(self.dataloader)} batches")
        
    def get_callbacks(self):
        """Tạo callbacks cho training - ĐƠN GIẢN"""
        checkpoint_callback = ModelCheckpoint(
            dirpath=CHECKPOINT_DIR,
            filename="football-chatbot-{epoch:02d}-{train_loss:.2f}",
            monitor="train_loss",
            mode="min",
            save_top_k=2,
            save_last=True,
            every_n_epochs=1
        )
        
        early_stop_callback = EarlyStopping(
            monitor="train_loss",
            min_delta=0.001,
            patience=10,  # Giảm patience
            mode="min",
            verbose=True
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        return [checkpoint_callback, early_stop_callback, lr_monitor]
    
    def get_logger(self):
        """Tạo logger cho training"""
        return TensorBoardLogger(
            save_dir=LOG_DIR,
            name="football_chatbot_simple",
            version="v1.0"
        )
    
    def train(self):
        """Thực hiện training - ĐƠN GIẢN"""
        self.setup_training()
        
        print("🚀 Bắt đầu training với cấu hình đơn giản...")
        print(f"📊 Số lượng parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        trainer = L.Trainer(
            max_epochs=EPOCHS,
            accelerator="auto",
            devices="auto",
            logger=self.get_logger(),
            callbacks=self.get_callbacks(),
            accumulate_grad_batches=1,  # Không tích lũy gradient
            gradient_clip_val=0.5,      # Gradient clipping nhỏ hơn
            precision="32-true",        # Dùng precision 32-bit cho ổn định
            log_every_n_steps=10,
            enable_progress_bar=True,
            overfit_batches=0,
            deterministic=False
        )
        
        try:
            # Training
            trainer.fit(self.model, train_dataloaders=self.dataloader)
            
            # Lưu mô hình cuối cùng
            self.save_final_model()
            
            print("✅ Training hoàn tất!")
            
        except Exception as e:
            print(f"❌ Lỗi trong quá trình training: {e}")
            raise
    
    def save_final_model(self):
        """Lưu mô hình đã trained"""
        model_save_data = {
            'model_state_dict': self.model.state_dict(),
            'vocab_size': len(self.vocab),
            'd_model': D_MODEL,
            'max_len': MAX_LEN,
            'vocab': {
                'token2id': self.vocab.token2id,
                'id2token': self.vocab.id2token
            },
            'training_config': {
                'epochs': EPOCHS,
                'learning_rate': LEARNING_RATE,
                'batch_size': BATCH_SIZE
            }
        }
        
        torch.save(model_save_data, MODEL_SAVE_PATH)
        print(f"💾 Mô hình đã được lưu tại: {MODEL_SAVE_PATH}")

def main():
    """Hàm main để chạy training"""
    try:
        trainer = SimpleTrainer()
        trainer.train()
    except Exception as e:
        print(f"❌ Lỗi trong quá trình training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()