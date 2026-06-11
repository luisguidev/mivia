import cv2
import time
import torch
from src.pipeline.ingestion import VideoBuffer
from src.engine.model import ActionRecognizer

def main():
    # 1. Configurações Globais
    VIDEO_SOURCE = "rtsp://mivia-cam:mivia-camppac04@192.168.68.114:554/stream1"  
    CONFIDENCE_THRESHOLD = 0.80 
    
    print("[INFO] Iniciando o pipeline...")
    
    try:
        reconhecedor = ActionRecognizer()
        buffer = VideoBuffer(source=VIDEO_SOURCE, window_size=16)
        
        # Ajuste de buffer para latência mínima
        buffer.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
    except Exception as e:
        print(f"[ERRO CRÍTICO] Falha na inicialização: {e}")
        return

    print("[INFO] Sistema Operacional. Pressione 'q' para encerrar.")
    
    ultima_acao = "Analisando..."
    ultima_confianca = 0.0
    
    # 2. Loop Principal Protegido
    try:
        while True:
            start_time = time.time()
            
            sucesso, clip_temporal = buffer.process_next_frame()
            if not sucesso:
                print("[INFO] Fim da transmissão.")
                break
            
            # Pega frame para exibição
            frame_display = buffer.cap.read()[1] if buffer.cap.isOpened() else None
            
            if clip_temporal is not None:
                infer_start = time.time()
                acao, confianca = reconhecedor.predict(clip_temporal)
                infer_time = (time.time() - infer_start) * 1000
                
                if confianca >= CONFIDENCE_THRESHOLD:
                    ultima_acao = acao
                    ultima_confianca = confianca
                    if "dropping" in acao.lower() or "throwing" in acao.lower():
                        print(f"[ALERTA] Descarte Detectado! Confiança: {confianca*100:.1f}%")
                
                print(f"Inferência: {infer_time:.1f}ms | Ação: {acao}")

            if frame_display is not None:
                cv2.putText(frame_display, f"Acao: {ultima_acao}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame_display, f"Conf: {ultima_confianca*100:.1f}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Monitoramento - MIVIA/UFC", frame_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("[INFO] Interrupção detectada.")
    finally:
        # 3. Limpeza Garantida
        buffer.release()
        cv2.destroyAllWindows()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("[INFO] Memória liberada.")

if __name__ == "__main__":
    main()