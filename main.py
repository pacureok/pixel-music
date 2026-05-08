import torch
import soundfile as sf
import argparse
import os
from diffusers import StableAudioPipeline
from huggingface_hub import login

def PIXEL_MU():
    parser = argparse.ArgumentParser(description="Generador PIXEL-MU PRO")
    parser.add_argument("prompt", type=str, help="Descripción detallada de la música")
    parser.add_argument("-N", "--negative", type=str, default="low quality, mono, distorted, hiss, noise, vocals", help="Lo que queremos evitar")
    parser.add_argument("-T", "--token", type=str, required=True, help="Token de Hugging Face")
    parser.add_argument("-D", "--duration", type=float, default=47.0, help="Duración en segundos (máx 47)")
    
    args = parser.parse_args()

    try:
        print("Iniciando sesión en Hugging Face...")
        login(token=args.token)

        # Cargar el modelo con máxima precisión
        model_id = "stabilityai/stable-audio-open-1.0"
        print(f"Cargando modelo {model_id}...")
        
        pipe = StableAudioPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")

        print(f"\n--- GENERANDO OBRA MAESTRA ---")
        print(f"Estilo: {args.prompt}")

        # Optimizamos los pasos a 250 para mayor detalle y claridad
        generator = torch.Generator("cuda").manual_seed(torch.seed()) # Semilla aleatoria cada vez
        
        audio = pipe(
            args.prompt,
            negative_prompt=args.negative,
            num_inference_steps=250,      # Calidad superior
            audio_end_in_s=args.duration, # Máximo permitido
            guidance_scale=15.0,          # Control de estilo más fuerte
            num_waveforms_per_prompt=1,
            generator=generator,
        ).audios

        output_file = "pixel_pro_music.wav"
        output = audio[0].T.float().cpu().numpy()
        sf.write(output_file, output, pipe.vae.sampling_rate)
        
        print(f"\nGeneración completada. Archivo: {output_file}")

    except Exception as e:
        print(f"\n[ERROR]: {e}")

if __name__ == "__main__":
    PIXEL_MU()
