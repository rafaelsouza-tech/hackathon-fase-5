from ultralytics import YOLO
import shutil
import os

# Caminho do modelo pré-treinado dentro da pasta "models"
pretrained_model_path = "models/yolov8n.pt"

# Caminho final onde o modelo treinado será salvo
final_model_path = "models/best.pt"

# Garante que a pasta "models/" exista
os.makedirs("models", exist_ok=True)

# Carregar o modelo YOLOv8 pré-treinado
model = YOLO(pretrained_model_path)  # Agora pegando de models/

# Treinar o modelo com seu dataset personalizado
model.train(
    data="dataset/data.yaml",  # Caminho para o arquivo de configuração do dataset
    epochs=50,  # Número de épocas de treinamento
    batch=16,  # Tamanho do batch (ajuste conforme sua GPU)
    imgsz=640,  # Tamanho das imagens usadas no treinamento
    workers=4,  # Número de threads para carregamento de dados
    device="cuda",  # Usa GPU se disponível, senão usa CPU
    project="models",  # Define onde os arquivos serão salvos
    name="train_run",  # Nome da pasta de saída dentro de "models/"
    exist_ok=True  # Reutiliza a pasta sem sobrescrever conteúdos antigos
)

# Caminho do modelo gerado automaticamente pelo YOLO
trained_model_path = "models/train_run/weights/best.pt"

# Move o modelo treinado para models/best.pt
shutil.move(trained_model_path, final_model_path)

print(f"Treinamento concluído! O modelo treinado foi salvo em '{final_model_path}'")