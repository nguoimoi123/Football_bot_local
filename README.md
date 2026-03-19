# Football_bot_local

# 💻 Tech Stack

### Programming Languages
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)

### AI / Machine Learning
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

### Data Science
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

### Backend
![Django](https://img.shields.io/badge/django-%23092E20.svg?style=for-the-badge&logo=django&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)

### Web
![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![Vite](https://img.shields.io/badge/vite-%23646CFF.svg?style=for-the-badge&logo=vite&logoColor=white)

### Deployment
![Vercel](https://img.shields.io/badge/vercel-%23000000.svg?style=for-the-badge&logo=vercel&logoColor=white)
![Render](https://img.shields.io/badge/Render-%46E3B7.svg?style=for-the-badge&logo=render&logoColor=white)

### Databases
![MongoDB](https://img.shields.io/badge/MongoDB-%234ea94b.svg?style=for-the-badge&logo=mongodb&logoColor=white)
![MySQL](https://img.shields.io/badge/mysql-4479A1.svg?style=for-the-badge&logo=mysql&logoColor=white)

### GPU Computing
![CUDA](https://img.shields.io/badge/cuda-000000.svg?style=for-the-badge&logo=nVIDIA&logoColor=green)

---

## Overview

Football_bot_local is an offline chatbot based on PyTorch/NLP for football conversation and question-answering.

## Repository Structure

- `chat.py`: Main script to run the chatbot interaction loop.
- `config.py`: Model and training configuration settings.
- `data.py`: Preprocessing and dataset handling.
- `model.py`: Model definition (e.g., sequence-to-sequence, transformer).
- `train.py`: Training pipeline and checkpointing.
- `vocabulary.json`, `data.txt`: Data artifacts used for vocab and corpus.
- `football_chatbot_optimized.pth`: Trained model weights.
- `checkpoints/`: Training checkpoint files.
- `logs/`: TensorBoard logs.

## Getting Started

1. Clone repository (already local):
   ```bash
   git clone https://github.com/nguoimoi123/Football_bot_local.git
   cd Football_bot_local
   ```

2. (Optional) Create and activate virtual environment:
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate   # Windows PowerShell
   # or source .venv/bin/activate  # Linux/Mac
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run training (if you want to re-train):
   ```bash
   python train.py
   ```

5. Run chatbot:
   ```bash
   python chat.py
   ```

## Notes

- If model weights are already present, `chat.py` can run immediately.
- For best performance, use a GPU with CUDA (if installed) and configure in `config.py`.

## License

MIT License
