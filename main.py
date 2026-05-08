import torch
import soundfile as sf
from diffusers import AudioLDM2Pipeline
import os

def generate_music():
    print("--- Iniciando Generador de Música Nivel Suno ---")
    
    # Usamos AudioLDM2-Music: No es Gated (Sin error 401) y es SOTA
    model_id = "cvssp/audioldm2-music"
    
    print(f"Cargando modelo en 2 GPUs T4...")
    
    # Cargamos con float16 para velocidad y repartimos en las GPUs
    pipe = AudioLDM2Pipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        device_map="balanced"
    )

    # Prompt tipo Suno
    prompt = "A high-energy electronic dance track with heavy bass, synth melodies, and a professional club atmosphere, 128 BPM"
    negative_prompt = "low quality, distortion, noise, monotone, vocals"

    print(f"Generando audio: '{prompt}'")
    
    # Generación
    # AudioLDM2 genera resultados muy limpios con 50-100 pasos
    with torch.inference_mode():
        audio = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=100,
            audio_length_in_s=30, # Duración en segundos
            num_waveforms_per_prompt=1
        ).audios[0]

    # Guardar resultado
    output_path = "resultado_suno_clone.wav"
    sf.write(output_path, audio, 48000) # AudioLDM2 usa 48kHz
    
    print(f"--- ¡Éxito! Archivo guardado como: {output_path} ---")

if __name__ == "__main__":
    generate_music()
