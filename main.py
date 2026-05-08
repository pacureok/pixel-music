import torch
import soundfile as sf
import argparse
import os
from diffusers import StableAudioPipeline
from huggingface_hub import login

def run_pixel_music():
    # Configuración de argumentos para el comando personalizado
    parser = argparse.ArgumentParser(description="PIXEL-MU CLI")
    parser.add_argument("prompt", type=str, help="El prompt principal entre comillas")
    parser.add_argument("-YT", "--youtube", type=str, help="URL de YouTube de base (Opcional)", default=None)
    parser.add_argument("-N", "--negative", type=str, help="Lo que no debe hacer", default="Low quality, noise")
    
    args = parser.parse_args()

    # Autenticación (Reemplaza con tu token si es necesario)
    HF_TOKEN = "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"
    login(token=HF_TOKEN)

    # Si hay URL de YouTube, podrías usarla aquí (Stable Audio Open es solo Text-to-Audio)
    if args.youtube:
        print(f"--- Nota: Base de YouTube detectada ({args.youtube}). ---")
        print("El modelo actual solo usa el prompt de texto, pero la URL está lista para futuras expansiones.")

    print("--- Cargando Modelo PIXEL-MU ---")
    model_id = "stabilityai/stable-audio-open-1.0"
    
    pipe = StableAudioPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16
    ).to("cuda")

    print(f"Generando: {args.prompt}")
    print(f"Evitando: {args.negative}")

    with torch.inference_mode():
        generator = torch.Generator("cuda").manual_seed(42)
        output = pipe(
            args.prompt,
            negative_prompt=args.negative,
            num_inference_steps=200,
            audio_end_in_s=30.0,
            num_waveforms_per_prompt=1,
            generator=generator
        ).audios

    audio_data = output[0].T.float().cpu().numpy()
    sf.write("pixel_output.wav", audio_data, pipe.vae.sampling_rate)
    print("--- ¡Éxito! Archivo guardado como pixel_output.wav ---")

if __name__ == "__main__":
    run_pixel_music()
