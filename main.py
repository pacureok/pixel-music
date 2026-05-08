import torch
import soundfile as sf
from diffusers import StableAudioPipeline
from huggingface_hub import login
import os

def generate_music():
    # 1. Autenticación con tu Token
    # Nota: En un entorno real, es mejor usar variables de entorno
    HF_TOKEN = "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"
    login(token=HF_TOKEN)

    print("--- Cargando Stable Audio Open 1.0 (Stability AI) ---")
    
    model_id = "stabilityai/stable-audio-open-1.0"
    
    # Cargamos el pipeline en media precisión (float16) para no agotar la RAM de la T4
    pipe = StableAudioPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    # Configuramos los prompts
    prompt = "A professional tech house track, 128 BPM, high energy, punchy kick, melodic synth, club atmosphere"
    negative_prompt = "Low quality, static, noise, distorted, vocals, mono"

    print(f"Generando audio de alta fidelidad...")
    
    # Generación de audio
    # audio_end_in_s define la duración (máximo 47s para este modelo)
    with torch.inference_mode():
        generator = torch.Generator("cuda").manual_seed(42) # Semilla fija para consistencia
        output = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=200,
            audio_end_in_s=30.0, 
            num_waveforms_per_prompt=1,
            generator=generator
        ).audios

    # El modelo devuelve un tensor [batch, canales, muestras]
    # Lo transponemos para que soundfile lo entienda (muestras, canales)
    audio_data = output[0].T.float().cpu().numpy()
    
    output_filename = "stable_audio_output.wav"
    sf.write(output_filename, audio_data, pipe.vae.sampling_rate)
    
    print(f"--- ¡Éxito! Música guardada como: {output_filename} ---")

if __name__ == "__main__":
    generate_music()
