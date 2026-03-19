# Cấu hình nâng cao cho Football Chatbot
import os

# ---- SIÊU THAM SỐ TỐI ƯU ----
D_MODEL = 256        # Giảm để training nhanh hơn
MAX_LEN = 80         # Độ dài chuỗi tối ưu
BATCH_SIZE = 16      # Giảm batch size để tránh memory issues
LEARNING_RATE = 0.0001
EPOCHS = 50          # Giảm epochs
NUM_HEADS = 4        # Giảm số heads
NUM_LAYERS = 2       # Giảm số layers
DROPOUT_RATE = 0.1   # Giảm dropout


# ---- ĐƯỜNG DẪN ----
MODEL_SAVE_PATH = "football_chatbot_optimized.pth"
CHECKPOINT_DIR = "checkpoints/"
LOG_DIR = "logs/"

# ---- CẤU HÌNH GENERATION ----
MAX_GENERATION_LENGTH = 50
TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.9

# ---- PRE-TRAINED MODEL ----
VIETNAMESE_SBERT_MODEL = "keepitreal/vietnamese-sbert"

# ---- TỐI ƯU HÓA ----
ACCUMULATE_GRAD_BATCHES = 2
PATIENCE_EARLY_STOPPING = 15
USE_AMP = True  # Automatic Mixed Precision

# Tạo thư mục nếu chưa tồn tại
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)