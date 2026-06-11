import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

class ActionRecognizer:
    def __init__(self, model_ckpt="MCG-NJU/videomae-base-finetuned-kinetics"):
        """
        Inicializa o processador e aloca o modelo na GPU usando precisão otimizada.
        """
        # 1. Definição do Hardware
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 2. Carregamento do Processador (A interface matemática)
        self.processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
        
        # 3. Carregamento do Modelo com otimização Float16
        self.model = VideoMAEForVideoClassification.from_pretrained(
            model_ckpt,
            torch_dtype=torch.float16,  # Metade do consumo de VRAM
            low_cpu_mem_usage=True      # Evita picos de RAM durante o carregamento
        )
        
        # 4. Envio dos pesos para a Placa de Vídeo e modo de avaliação
        self.model.to(self.device)
        self.model.eval()

    def predict(self, clip_temporal):
        """
        Recebe a matriz (16, 224, 224, 3) do buffer, envia para a rede e 
        retorna a ação detectada e sua confiança.
        """
        # Transformamos a matriz NumPy 4D em uma lista de frames individuais 
        # (formato exigido pelo processador do HuggingFace)
        frames = list(clip_temporal)

        # 1. O Pré-processamento
        inputs = self.processor(frames, return_tensors="pt")

        # 2. Envio dos dados para a GPU na mesma precisão do modelo
        pixel_values = inputs["pixel_values"].to(self.device, dtype=torch.float16)

        # 3. O Escudo de Memória (Inferência pura)
        with torch.no_grad():
            outputs = self.model(pixel_values)
            logits = outputs.logits

        # 4. Tradução Matemática (Softmax)
        probabilidades = torch.nn.functional.softmax(logits, dim=-1)
        
        # 5. Extração do resultado com maior confiança
        confianca_maxima, classe_id = torch.max(probabilidades, dim=-1)
        
        # Traduzimos o ID numérico para o nome da classe (em texto)
        nome_acao = self.model.config.id2label[classe_id.item()]
        
        return nome_acao, confianca_maxima.item()