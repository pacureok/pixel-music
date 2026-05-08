import torch
import soundfile as sf
import argparse
import os
from diffusers import StableAudioPipeline
from huggingface_hub import login

def PIXEL_MU():
    parser = argparse.ArgumentParser(description="Generador de Audio PIXEL-MU")
    parser.add_argument("prompt", type=str, help="Descripción del audio")
    parser.add_argument("-YT", "--youtube", type=str, default=None, help="URL de YouTube")
    parser.add_argument("-N", "--negative", type=str, default="Low quality", help="Lo que no quieres")
    
    args = parser.parse_args()

    # --- REEMPLAZA ESTO CON TU NUEVO TOKEN ---
    HF_TOKEN = "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" 
    # -----------------------------------------

    try:
        if HF_TOKEN == "TU_NUEVO_TOKEN_AQUÍ":
            print("Error: Debes poner tu nuevo token de Hugging Face en el script.")
            return

        login(token=HF_TOKEN)

        model_id = "stabilityai/stable-audio-open-1.0"
        print(f"\nComponiendo: {args.prompt}")

        # Configuración según documentación oficial de Diffusers
        pipe = StableAudioPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        pipe = pipe.to("cuda")

        generator = torch.Generator("cuda").manual_seed(0)

        audio = pipe(
            args.prompt,
            negative_prompt=args.negative,
            num_inference_steps=200,
            audio_end_in_s=30.0, # Generamos 30 segundos
            num_waveforms_per_prompt=1,
            generator=generator,
        ).audios

        output_file = "pixel_music_result.wav"
        output = audio[0].T.float().cpu().numpy()
        sf.write(output_file, output, pipe.vae.sampling_rate)
        
        print(f"¡Éxito! Audio guardado en {output_file}")

    except Exception as e:
        print(f"Hubo un problema: {e}")

if __name__ == "__main__":
    PIXEL_MU()
