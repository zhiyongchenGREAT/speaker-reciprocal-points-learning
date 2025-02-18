import os
import torchaudio
import torch
import pyroomacoustics as pra
import numpy as np
from tqdm import tqdm  
import librosa
import random

audio_dir = "/server9/speech_group/wsh/datasets/SHAL_sovits_correct_reorganized"
output_dir = "/server9/speech_group/wsh/datasets/SHAL_sovits_correct_aug/wav_aug_smallroom"

def generate_noise_with_snr(waveform, snr_db):
    signal_power = torch.mean(waveform ** 2)
    
    snr_linear = 10 ** (snr_db / 10)  
    noise_power = signal_power / snr_linear 
    noise = torch.randn(waveform.size())
    noise = noise * torch.sqrt(noise_power / torch.mean(noise ** 2))
    noisy_waveform = waveform + noise
    return noisy_waveform

def change_speed_without_pitch_shift(waveform, sample_rate, speed_factor=1.0):
    stretched_waveform = librosa.effects.time_stretch(waveform.numpy(), rate=speed_factor)
    return torch.tensor(stretched_waveform)

def small_room_dimensions():
    return [10, 7, 4]  

def near_field_mic_and_src_positions(room_dim):
    src_loc = [5, 3, 2]  
    near_mic_loc = [6, 4, 2]
    return near_mic_loc, src_loc

def add_reverb_room(waveform, sample_rate, room_dim, mic_loc, src_loc):
    max_order = 15 
    absorption = 0.01  

    room = pra.ShoeBox(room_dim, fs=sample_rate, max_order=max_order, absorption=absorption)
    room.add_source(src_loc, signal=waveform.numpy().flatten())
    room.add_microphone_array(pra.MicrophoneArray(np.array([mic_loc]).T, room.fs))
    room.simulate()

    reverb_waveform = torch.tensor(room.mic_array.signals[0]).unsqueeze(0)
    return reverb_waveform

def apply_volume_perturbation(waveform):
    volume_factor = random.uniform(0.8, 1.2) 
    return waveform * volume_factor

def apply_local_rhythm_jitter(waveform, sample_rate, segment_duration=0.2):
    num_samples = waveform.shape[1]
    segment_length = int(segment_duration * sample_rate)
    segments = [waveform[:, i:i + segment_length] for i in range(0, num_samples, segment_length)]
    
    augmented_segments = []
    for segment in segments:
        if segment.shape[1] > 0:
            jitter_factor = random.uniform(0.5, 1.5)
            augmented_segment = change_speed_without_pitch_shift(segment, sample_rate, jitter_factor)
            augmented_segments.append(augmented_segment)
    
    augmented_waveform = torch.cat(augmented_segments, dim=1)
    return augmented_waveform

def apply_global_speed_change(waveform, sample_rate):
    speed_factor = random.uniform(0.3, 2.0)
    return change_speed_without_pitch_shift(waveform, sample_rate, speed_factor)

def augment_and_save_audio(input_path, output_path):
    waveform, sample_rate = torchaudio.load(input_path)

    snr_db = 15
    waveform = generate_noise_with_snr(waveform, snr_db)
    waveform = apply_volume_perturbation(waveform)
    waveform = apply_local_rhythm_jitter(waveform, sample_rate, segment_duration=0.5)
    waveform = apply_global_speed_change(waveform, sample_rate)

    room_dim = small_room_dimensions()
    mic_loc, src_loc = near_field_mic_and_src_positions(room_dim)
    waveform = add_reverb_room(waveform, sample_rate, room_dim, mic_loc, src_loc)

    output_path = os.path.join(output_path, os.path.basename(input_path))
    torchaudio.save(output_path, waveform, sample_rate)

for speaker_dir in tqdm(os.listdir(audio_dir), desc="Processing Speakers"):
    speaker_dir_path = os.path.join(audio_dir, speaker_dir)
    output_speaker_dir = os.path.join(output_dir, speaker_dir)
    os.makedirs(output_speaker_dir, exist_ok=True)

    audio_files = [f for f in os.listdir(speaker_dir_path) if f.endswith(".wav")]
    
    for audio_file in tqdm(audio_files, desc=f"Processing {speaker_dir}", leave=False):
        input_path = os.path.join(speaker_dir_path, audio_file)
        augment_and_save_audio(input_path, output_speaker_dir)

print("Audio augmentation with near field small room completed and files saved.")
