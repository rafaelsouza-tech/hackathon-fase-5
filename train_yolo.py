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
    data="datasets/dataset/data.yaml",  # Caminho do dataset
    epochs=50,  # Número de épocas de treinamento
    batch=16,  # Tamanho do batch
    imgsz=640,  # Tamanho da imagem
    workers=4,  # Número de processos para carregamento de dados
    device="cuda",  # Treina na GPU (se disponível)
    project="models",  # Salva os resultados dentro de "models/"
    name="train_run",  # Nome do treino dentro de "models/"
    exist_ok=True,  # Permite reutilizar a pasta
    pretrained=True,  # Mantém o aprendizado do modelo original
    optimizer="SGD",  # Usa o otimizador SGD
    amp=True,  # Ativa precisão mista para treinar mais rápido
    resume=False,  # NÃO tentar carregar um checkpoint antigo
    close_mosaic=10,  # Ajusta mosaico para estabilizar treino
    cos_lr=True,  # Usa decaimento de taxa de aprendizado em formato cosseno
    lr0=0.01,  # Taxa de aprendizado inicial
    weight_decay=0.0005,  # Decaimento de peso (regularização)
    momentum=0.937,  # Momento do SGD
    warmup_epochs=3.0,  # Quantidade de épocas de warmup
    val=True,  # Ativa validação durante o treino
    save=True  # Salva checkpoints do melhor modelo
)

# Caminho do modelo gerado automaticamente pelo YOLO
trained_model_path = "models/train_run/weights/best.pt"

# Move o modelo treinado para models/best.pt
shutil.move(trained_model_path, final_model_path)

print(f"Treinamento concluído! O modelo treinado foi salvo em '{final_model_path}'")