import cv2
import time
from src.pipeline.ingestion import VideoBuffer
from src.engine.model import ActionRecognizer

def main():
    # 1. Configurações Globais
    # Para testar antes do dataset chegar, você pode colocar 0 para usar sua webcam
    VIDEO_SOURCE = "rtsp://mivia-cam:mivia-camppac04@192.168.68.114:554/stream1"  
    CONFIDENCE_THRESHOLD = 0.80  # 80% de certeza mínima
    
    print("[INFO] Iniciando o pipeline de Videomonitoramento...")
    print(f"[INFO] Fonte de vídeo: {VIDEO_SOURCE}")
    
    # 2. Instanciação dos Módulos
    # Isso aloca a memória da GPU e conecta na câmera
    try:
        reconhecedor = ActionRecognizer()
        buffer = VideoBuffer(source=VIDEO_SOURCE, window_size=16)
    except Exception as e:
        print(f"[ERRO CRÍTICO] Falha na inicialização: {e}")
        return

    print("[INFO] Sistema Operacional. Pressione 'q' para encerrar.")
    
    # Variáveis de estado para a interface gráfica
    ultima_acao = "Analisando..."
    ultima_confianca = 0.0
    
    # 3. O Loop Infinito (O Coração do Sistema)
    while True:
        start_time = time.time()
        
        # Puxa o próximo frame da câmera
        sucesso, clip_temporal = buffer.process_next_frame()
        
        if not sucesso:
            print("[INFO] Fim da transmissão de vídeo.")
            break
            
        # Pega o frame atual bruto (sem resize) direto do OpenCV para mostrar na tela
        frame_display = buffer.cap.read()[1] if buffer.cap.isOpened() else None
        
        # Se o buffer encheu e devolveu um clipe de 16 frames, rodamos a IA
        if clip_temporal is not None:
            infer_start = time.time()
            
            # A Magia acontece aqui
            acao, confianca = reconhecedor.predict(clip_temporal)
            
            infer_time = (time.time() - infer_start) * 1000 # em milissegundos
            
            # Atualiza o estado visual apenas se passar do nosso limiar
            if confianca >= CONFIDENCE_THRESHOLD:
                ultima_acao = acao
                ultima_confianca = confianca
                
                # Se a ação detectada for a do nosso interesse, aqui seria o local 
                # para inserir a requisição HTTP enviando o alerta para um banco de dados.
                if "dropping" in acao.lower() or "throwing" in acao.lower():
                    print(f"[ALERTA] Descarte Detectado! Confiança: {confianca*100:.1f}%")
            
            print(f"Tempo de Inferência GPU: {infer_time:.1f}ms | Ação: {acao} ({confianca:.2f})")

        # 4. Feedback Visual (Interface de Debug)
        if frame_display is not None:
            # Desenha um retângulo e o texto da predição na tela
            cv2.putText(frame_display, f"Acao: {ultima_acao}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_display, f"Conf: {ultima_confianca*100:.1f}%", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Calcula e exibe o FPS total do loop
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(frame_display, f"FPS: {fps:.1f}", (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.imshow("Monitoramento de Descarte - MIVIA/UFC", frame_display)

        # Condição de saída: Se apertar a tecla 'q', quebra o loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Encerrando o sistema pelo usuário.")
            break

    # 5. Limpeza de Memória
    buffer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()