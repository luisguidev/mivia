import cv2
import numpy as np
from collections import deque

class VideoBuffer:
    def __init__(self, source, window_size=16, target_size=(224, 224)):
        """
        Inicializa o pipeline de captura e a fila de memória FIFO.
        """
        self.source = source
        self.window_size = window_size
        self.target_size = target_size
        
        # O coração da janela deslizante: uma fila com tamanho máximo fixo
        self.buffer = deque(maxlen=self.window_size)
        
        # Conexão com a câmera ou arquivo de vídeo
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Erro ao abrir a fonte de vídeo: {self.source}")

    def process_next_frame(self):
        """
        Lê o próximo frame, aplica o pré-processamento e empurra para a fila.
        Retorna (sucesso, clipe_pronto)
        """
        ret, frame = self.cap.read()
        
        # Se 'ret' for False, o vídeo acabou ou a câmera caiu
        if not ret:
            return False, None
            
        # 1. Redimensionamento Espacial (Exigência do VideoMAE)
        frame_resized = cv2.resize(frame, self.target_size)
        
        # 2. Correção de Cores (De BGR do OpenCV para RGB do PyTorch)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # 3. Empurra para a Fila FIFO
        self.buffer.append(frame_rgb)
        
        # 4. Verifica se já temos frames suficientes para formar um "clipe" temporal
        if len(self.buffer) == self.window_size:
            # Converte a fila em uma matriz NumPy contígua na memória
            # Formato de saída: (Tempo, Altura, Largura, Canais)
            clip = np.array(self.buffer)
            return True, clip
            
        # Se a fila ainda está enchendo (ex: frame 5 de 16), retorna None
        return True, None

    def release(self):
        """
        Libera o hardware da câmera e destrói os ponteiros de memória.
        """
        self.cap.release()