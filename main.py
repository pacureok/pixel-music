import torch
import soundfile as sf
import argparse
import sys
import os
from diffusers import StableAudioPipeline
from huggingface_hub import login

def PIXEL_MU():
    # Configuración de comandos: PIXEL-MU("prompt") -YT ("url") -N ("negativo")
    parser = argparse.ArgumentParser(description="Generador de Audio PIXEL-MU")
    parser.add_argument("prompt", type=str, help="Descripción del audio a generar")
    parser.add_argument("-YT", "--youtube", type=str, default=None, help="URL de base de YouTube")
    parser.add_argument("-N", "--negative", type=str, default="Low quality, noise", help="Lo que no quieres en el audio")
    
    args = parser.parse_args()

    # Tu Token de Hugging Face
    HF_TOKEN = "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"
    
    try:
        login(token=HF_TOKEN)

        print("\n--- INICIANDO PROCESO PIXEL-MU ---")
        print(f"PROMPT: {args.prompt}")
        print(f"EVITAR: {args.negative}")
        if args.youtube:
            print(f"INFO: Se ha recibido base de YouTube: {args.youtube}")

        model_id = "stabilityai/stable-audio-open-1.0"
        
        # Carga optimizada para la GPU T4 de Kaggle
        print("Cargando modelo en memoria GPU...")
        pipe = StableAudioPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16
        ).to("cuda")

        print("Generando audio de alta fidelidad...")
        with torch.inference_mode():
            # Semilla aleatoria para variar resultados, o fija para consistencia
            seed = torch.randint(0, 1000000, (1,)).item()
            generator = torch.Generator("cuda").manual_seed(seed)
            
            output = pipe(
                args.prompt,
                negative_prompt=args.negative,
                num_inference_steps=200,
                audio_end_in_s=30.0, 
                num_waveforms_per_prompt=1,
                generator=generator
            ).audios

        # Guardar resultado
        audio_data = output[0].T.float().cpu().numpy()
        output_filename = "pixel_music_output.wav"
        sf.write(output_filename, audio_data, pipe.vae.sampling_rate)
        
        print(f"\n--- COMPLETADO ---")
        print(f"Resultado guardado en: /kaggle/working/pixel-music/{output_filename}")

    except Exception as e:
        print(f"\n[ERROR CRÍTICO]: {e}")

if __name__ == "__main__":
    PIXEL_MU()
