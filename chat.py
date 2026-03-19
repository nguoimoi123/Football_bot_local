# Chat interface nâng cao với nhiều tính năng
import torch
import random
import re
from underthesea import word_tokenize
from model import FootballChatbot
from data import Vocabulary, load_vocabulary
from config import *
from sentence_transformers import SentenceTransformer, util

class EnhancedFootballChatbot:
    def __init__(self, model_path=MODEL_SAVE_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.vocab = None
        self.semantic_model = None
        self.questions = []
        self.answers = []
        
        self.load_resources(model_path)
        self.setup_semantic_search()
        
    def load_resources(self, model_path):
        """Tải mô hình và resources"""
        try:
            # Tải vocabulary
            self.vocab = load_vocabulary("vocabulary.json")
            
            # Tải checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Khởi tạo mô hình
            self.model = EnhancedFootballChatbot(
                num_tokens=len(self.vocab),
                d_model=checkpoint['d_model'],
                max_len=checkpoint['max_len']
            )
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ Đã tải mô hình thành công!")
            
        except Exception as e:
            print(f"⚠️ Không thể tải mô hình: {e}")
            print("🔄 Sử dụng semantic search làm fallback...")
    
    def setup_semantic_search(self):
        """Thiết lập semantic search"""
        try:
            self.semantic_model = SentenceTransformer(VIETNAMESE_SBERT_MODEL)
            self.load_qa_pairs()
            self.question_embeddings = self.semantic_model.encode(
                self.questions, convert_to_tensor=True
            )
            print("✅ Đã thiết lập semantic search!")
        except Exception as e:
            print(f"⚠️ Lỗi semantic search: {e}")
    
    def load_qa_pairs(self, filepath="data.txt"):
        """Tải cặp Q&A cho semantic search"""
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if "<EOS>" in line:
                    parts = line.split("<EOS>")
                    if len(parts) >= 2:
                        self.questions.append(parts[0].strip())
                        self.answers.append(parts[1].strip())
    
    def preprocess_input(self, text):
        """Tiền xử lý input"""
        # Chuẩn hóa text
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)  # Xóa punctuation
        return text
    
    def generate_response(self, input_text):
        """Sinh câu trả lời bằng mô hình"""
        try:
            # Encode input
            input_ids = self.vocab.encode(input_text, add_special_tokens=True)
            input_tensor = torch.tensor([input_ids], device=self.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(input_tensor)
                response_ids = generated_ids[0].cpu().tolist()
            
            # Decode response
            response = self.vocab.decode(response_ids)
            return self.postprocess_response(response)
            
        except Exception as e:
            print(f"⚠️ Lỗi generation: {e}")
            return None
    
    def semantic_search_response(self, input_text):
        """Tìm câu trả lời bằng semantic search"""
        try:
            query_embedding = self.semantic_model.encode(input_text, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, self.question_embeddings)[0]
            best_idx = torch.argmax(cos_scores).item()
            
            return self.answers[best_idx], self.questions[best_idx]
        except:
            return None, None
    
    def postprocess_response(self, response):
        """Hậu xử lý response"""
        # Làm sạch response
        response = re.sub(r'\s+', ' ', response).strip()
        
        # Thêm cảm xúc ngẫu nhiên
        emotions = ["⚽", "😊", "🔥", "💪", "🌟", "🎯"]
        if random.random() < 0.3:
            response += f" {random.choice(emotions)}"
        
        return response
    
    def get_response(self, user_input):
        """Lấy response cho input của user"""
        # Tiền xử lý
        processed_input = self.preprocess_input(user_input)
        
        # Thử generation trước
        if self.model:
            response = self.generate_response(processed_input)
            if response and len(response) > 5:  # Response hợp lệ
                return response, "generation"
        
        # Fallback về semantic search
        response, matched_question = self.semantic_search_response(processed_input)
        if response:
            return response, f"semantic (khớp: '{matched_question}')"
        
        # Fallback cuối cùng
        fallback_responses = [
            "Tôi chưa hiểu câu hỏi của bạn. Bạn có thể hỏi về bóng đá không? ⚽",
            "Xin lỗi, tôi chuyên về bóng đá. Bạn muốn biết gì về môn thể thao này?",
            "Hiện tôi chưa có thông tin cho câu hỏi này. Bạn hỏi về cầu thủ hay giải đấu nhé! 😊"
        ]
        return random.choice(fallback_responses), "fallback"

class ChatInterface:
    def __init__(self):
        self.chatbot = EnhancedFootballChatbot()
        self.conversation_history = []
        
    def display_welcome(self):
        """Hiển thị welcome message"""
        welcome_messages = [
            "⚽ Chào mừng đến với FootballBot Enhanced!",
            "🤖 Tôi là trợ lý bóng đá thông minh của bạn",
            "💡 Bạn có thể hỏi về: cầu thủ, giải đấu, đội bóng, thống kê...",
            "🔍 Tôi có thể trả lời bằng AI generation hoặc tìm kiếm thông minh",
            "📝 Gõ 'thoát' để kết thúc cuộc trò chuyện\n"
        ]
        
        for msg in welcome_messages:
            print(msg)
    
    def handle_special_commands(self, user_input):
        """Xử lý các lệnh đặc biệt"""
        special_commands = {
            "lịch sử": self.show_history,
            "xóa lịch sử": self.clear_history,
            "trợ giúp": self.show_help,
            "thoát": lambda: True
        }
        
        for cmd, handler in special_commands.items():
            if cmd in user_input.lower():
                return handler()
        
        return False
    
    def show_history(self):
        """Hiển thị lịch sử chat"""
        if not self.conversation_history:
            print("🤖 Chưa có lịch sử trò chuyện")
        else:
            print("\n📜 Lịch sử trò chuyện:")
            for i, (user, bot, method) in enumerate(self.conversation_history[-5:], 1):
                print(f"{i}. Bạn: {user}")
                print(f"   Bot: {bot} [{method}]")
        return False
    
    def clear_history(self):
        """Xóa lịch sử"""
        self.conversation_history.clear()
        print("🤖 Đã xóa lịch sử trò chuyện")
        return False
    
    def show_help(self):
        """Hiển thị trợ giúp"""
        help_text = """
🤖 **TRỢ GIÚP FOOTBALLBOT**

**Câu hỏi mẫu:**
- Ai là cầu thủ ghi nhiều bàn nhất?
- Manchester United có những danh hiệu gì?
- World Cup 2022 diễn ra ở đâu?
- So sánh Ronaldo và Messi

**Lệnh đặc biệt:**
- 'lịch sử' - Xem lịch sử chat
- 'xóa lịch sử' - Xóa lịch sử
- 'trợ giúp' - Hiển thị trợ giúp
- 'thoát' - Kết thúc

**Chế độ trả lời:**
- 🧠 Generation: Sử dụng AI để sinh câu trả lời
- 🔍 Semantic: Tìm câu trả lời gần nhất
- 🆘 Fallback: Câu trả lời dự phòng
        """
        print(help_text)
        return False
    
    def start_chat(self):
        """Bắt đầu chat interface"""
        self.display_welcome()
        
        while True:
            try:
                user_input = input("\n👤 Bạn: ").strip()
                
                if not user_input:
                    continue
                
                # Kiểm tra lệnh đặc biệt
                if self.handle_special_commands(user_input):
                    break
                
                # Lấy response
                response, method = self.chatbot.get_response(user_input)
                
                # Hiển thị response
                print(f"🤖 Bot: {response}")
                if method != "fallback":
                    print(f"   (Phương thức: {method})")
                
                # Lưu vào lịch sử
                self.conversation_history.append((user_input, response, method))
                
            except KeyboardInterrupt:
                print("\n\n🤖 Tạm biệt! Hẹn gặp lại ⚽")
                break
            except Exception as e:
                print(f"❌ Lỗi: {e}")
                print("🔄 Vui lòng thử lại...")

def main():
    """Hàm main để chạy chat"""
    try:
        chat_interface = ChatInterface()
        chat_interface.start_chat()
    except Exception as e:
        print(f"❌ Lỗi khởi động chatbot: {e}")

if __name__ == "__main__":
    main()