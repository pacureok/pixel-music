import torch
import soundfile as sf
from diffusers import StableAudioPipeline
from accelerate import init_empty_weights

def main():
    print("Iniciando motor musical en GPUs...")
    
    # El truco para usar T4 x2: device_map="auto" reparte el modelo en ambas GPUs
    # torch.float16 ahorra la mitad de la memoria RAM de video (VRAM)
    model_id = "stabilityai/stable-audio-open-1.0"
    
    print(f"Cargando modelo: {model_id} en múltiples GPUs...")
    pipe = StableAudioPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        device_map="auto" # <--- ESTO ES VITAL PARA USAR LAS 2 GPUs T4
    )

    # Prompt al estilo Suno
    prompt = "Un solo de guitarra eléctrica épico de heavy metal, batería rápida y agresiva, calidad de estudio"
    
    # Parámetros de generación
    seconds = 30 # Duración de la pista
    steps = 100  # Más pasos = mejor calidad, pero tarda más
    
    print(f"Generando audio para: '{prompt}'...")
    
    # Generar la música
    # generator fija una semilla para poder replicar la misma canción si quieres
    generator = torch.Generator(device="cuda").manual_seed(42)
    
    audio = pipe(
        prompt,
        audio_end_in_s=seconds,
        num_inference_steps=steps,
        generator=generator
    ).audios[0]

    # Guardar el archivo .wav
    output_filename = "mi_cancion_generada.wav"
    sample_rate = pipe.vae.sampling_rate
    
    # Convertir el tensor de PyTorch a un formato de audio guardable
    audio_np = audio.T.cpu().numpy()
    sf.write(output_filename, audio_np, sample_rate)
    
    print(f"¡Éxito! Canción guardada como {output_filename}")

if __name__ == "__main__":
    main()
