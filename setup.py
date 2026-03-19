# setup.py - Tự động thiết lập môi trường
import os
import sys

def check_data_file():
    """Kiểm tra và tạo file data.txt nếu chưa có"""
    if not os.path.exists("data.txt"):
        print("❌ Không tìm thấy file data.txt")
        print("📝 Đang tạo file data.txt mới...")
        
        # Nội dung mẫu cơ bản
        sample_data = """bạn thích đội bóng nào <EOS> tôi thích Manchester United, đội bóng có lịch sử vĩ đại ⚽ <EOS>
ai là vua phá lưới Premier League <EOS> Haaland đang dẫn đầu bảng xếp hạng ghi bàn 🔥 <EOS>
Real Madrid thắng bao nhiêu trận mùa này <EOS> họ đã thắng khoảng 28 trận rồi 💪 <EOS>
đội nào vô địch World Cup 2022 <EOS> Argentina đã vô địch, Messi tỏa sáng rực rỡ 🏆 <EOS>
bạn khỏe không <EOS> mình khỏe, cảm ơn bạn! luôn sẵn sàng nói chuyện bóng đá 😄 <EOS>"""
        
        with open("data.txt", "w", encoding="utf-8") as f:
            f.write(sample_data)
        
        print("✅ Đã tạo file data.txt với dữ liệu mẫu")
        print("💡 Hãy thêm dữ liệu thực vào file data.txt để training hiệu quả hơn")
    else:
        print("✅ Đã tìm thấy file data.txt")

def install_requirements():
    """Cài đặt dependencies"""
    print("📦 Đang cài đặt dependencies...")
    os.system("pip install -r requirements_enhanced.txt")

def main():
    """Thiết lập môi trường"""
    print("🔧 Thiết lập môi trường Football Chatbot Enhanced")
    
    # Kiểm tra data.txt
    check_data_file()
    
    # Cài đặt dependencies
    if input("Cài đặt dependencies? (y/n): ").lower() == 'y':
        install_requirements()
    
    print("\n✅ Thiết lập hoàn tất!")
    print("🚀 Bây giờ bạn có thể:")
    print("   - Chạy: python train_enhanced.py  (để training)")
    print("   - Chạy: python chat_enhanced.py   (để chat)")

if __name__ == "__main__":
    main()