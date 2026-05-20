from pathlib import Path
import torch
import note_seq


def decode_z_to_midi(z_parts, musicvae, controller, temperature=0.5, length=32):
    """
    Оригинальная функция decode_z_to_midi из Colab
    """
    with torch.no_grad():
        z_mv_hat = controller.decode_to_musicvae_z(z_parts)

    z_np = z_mv_hat.cpu().numpy()
    sequences = musicvae.decode(z_np, length=length, temperature=temperature)
    return sequences


def save_midi(sequence, output_path):
    """Сохраняет NoteSequence в MIDI файл"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    note_seq.sequence_proto_to_midi_file(sequence, str(output_path))
    print(f'  Saved: {output_path.name}')
