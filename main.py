import torch
import soundfile as sf
import argparse
import sys
from diffusers import StableAudioPipeline
from huggingface_hub import login

def PIXEL_MU():
    # 1. Configuración de Argumentos
    parser = argparse.ArgumentParser(description="Comando PIXEL-MU para generación de audio")
    parser.add_argument("prompt", type=str, help="Descripción de la música")
    parser.add_argument("-YT", "--youtube", type=str, default=None, help="URL de base de YouTube")
    parser.add_argument("-N", "--negative", type=str, default="Low quality, noise, distorted", help="Lo que no debe hacer")
    
    args = parser.parse_args()

    # 2. Autenticación (Usa tu token)
    HF_TOKEN = "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"
    login(token=HF_TOKEN)

    print(f"\n--- PIXEL-MU GENERATOR ---")
    print(f"PROMPT: {args.prompt}")
    print(f"NEGATIVO: {args.negative}")
    if args.youtube:
        print(f"BASE YT: {args.youtube} (Registrada para procesamiento)")

    # 3. Carga del Modelo
    print("\nCargando cerebro musical...")
    model_id = "stabilityai/stable-audio-open-1.0"
    
    try:
        pipe = StableAudioPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16
        ).to("cuda")

        # 4. Generación
        print("Componiendo audio (esto toma un momento)...")
        with torch.inference_mode():
            generator = torch.Generator("cuda").manual_seed(42)
            output = pipe(
                args.prompt,
                negative_prompt=args.negative,
                num_inference_steps=200,
                audio_end_in_s=30.0, # Duración de 30 segundos
                num_waveforms_per_prompt=1,
                generator=generator
            ).audios

        # 5. Guardado
        audio_data = output[0].T.float().cpu().numpy()
        output_file = "pixel_music_result.wav"
        sf.write(output_file, audio_data, pipe.vae.sampling_rate)
        
        print(f"\n--- ¡ÉXITO! ---")
        print(f"Archivo guardado como: {output_file}")

    except Exception as e:
        print(f"\n[ERROR]: {e}")

if __name__ == "__main__":
    PIXEL_MU()
