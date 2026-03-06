from pyannote.audio import Model
from pathlib import Path
from brouhaha.pipeline import RegressiveActivityDetectionPipeline
import torch 
from huggingface_hub import hf_hub_download
import numpy as np
import soundfile as sf
import io

model = None
ratio = 16000/270

def snr_apply(batch, rank=None, audio_column_name="audio", batch_size=32):
    global model
    if model is None:
        model = Model.from_pretrained(
            Path(hf_hub_download(repo_id="ylacombe/brouhaha-best", filename="best.ckpt")),
            strict=False,
        )
    if rank is not None or torch.cuda.device_count() > 0:
        # move the model to the right GPU if not there already
        device = f"cuda:{(rank or 0)% torch.cuda.device_count()}"
        # move to device and create pipeline here because the pipeline moves to the first GPU it finds anyway
        model.to(device)

    pipeline = RegressiveActivityDetectionPipeline(segmentation=model, batch_size = batch_size)
    if rank:
        pipeline.to(torch.device(device))
    
    device = pipeline._models["segmentation"].device

    def _ensure_sample_dict(sample):
        # Normalize various audio representations into a dict with 'array' and 'sampling_rate'
        # Acceptable inputs:
        # - dict with 'array' and 'sampling_rate'
        # - dict with 'path'
        # - path string
        # - raw bytes (wav)
        if isinstance(sample, dict):
            if "array" in sample and "sampling_rate" in sample:
                # ensure array is numpy and shaped (channels, time)
                arr = np.asarray(sample["array"])
                if arr.ndim == 1:
                    arr = arr[np.newaxis, :]
                elif arr.ndim == 2:
                    # soundfile and many decoders return (frames, channels).
                    # We want (channels, frames). Heuristic: if second dim looks
                    # like a small channel count (<=8), transpose.
                    if arr.shape[1] <= 8 and arr.shape[0] > arr.shape[1]:
                        arr = arr.T
                sample["array"] = arr.astype(np.float32)
                return sample
            if "path" in sample:
                path = sample["path"]
                data, sr = sf.read(path)
                arr = np.asarray(data)
                if arr.ndim == 1:
                    arr = arr[np.newaxis, :]
                elif arr.ndim == 2:
                    if arr.shape[1] <= 8 and arr.shape[0] > arr.shape[1]:
                        arr = arr.T
                return {"array": arr.astype(np.float32), "sampling_rate": sr}
        if isinstance(sample, (bytes, bytearray)):
            # read from bytes buffer
            bio = io.BytesIO(sample)
            data, sr = sf.read(bio)
            arr = np.asarray(data)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            elif arr.ndim == 2:
                if arr.shape[1] <= 8 and arr.shape[0] > arr.shape[1]:
                    arr = arr.T
            return {"array": arr.astype(np.float32), "sampling_rate": sr}
        if isinstance(sample, str):
            # treat as path-like
            data, sr = sf.read(sample)
            arr = np.asarray(data)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            elif arr.ndim == 2:
                if arr.shape[1] <= 8 and arr.shape[0] > arr.shape[1]:
                    arr = arr.T
            return {"array": arr.astype(np.float32), "sampling_rate": sr}
        # fallback: return as-is (may raise later)
        return sample

    def _to_waveform_tensor(arr, device):
        """Convert a numpy array (or array-like) to a torch tensor shaped (channels, time)."""
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        # If 1D, treat as (time,) -> mono (1, time)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        elif arr.ndim == 2:
            # Heuristic: prefer (channels, time). If shape looks like (frames, channels), transpose.
            frames, dim2 = arr.shape
            if frames > dim2 and dim2 <= 8:
                # likely (frames, channels)
                arr = arr.T
        else:
            # Collapse higher dimensions into time
            arr = arr.reshape(arr.shape[0], -1)
        # Ensure float32
        arr = arr.astype(np.float32)
        t = torch.from_numpy(arr).float()
        # Ensure 2D tensor
        if t.dim() == 1:
            t = t.unsqueeze(0)
        if t.dim() != 2:
            t = t.view(t.shape[0], -1)
        # Move to device
        try:
            t = t.to(device)
        except Exception:
            # If device move fails, return CPU tensor; pipeline can move it if needed
            pass
        return t

    val = batch[audio_column_name]
    # Detect batched input: lists/tuples or object-dtype numpy arrays are treated as batches
    is_batched = isinstance(val, (list, tuple)) or (isinstance(val, np.ndarray) and getattr(val, 'dtype', None) == object)
    if is_batched:
        snr = []
        c50 = []
        vad_durations = []
        for sample in val:
            sample = _ensure_sample_dict(sample)
            # sample['array'] is (channels, time) numpy array (or similar); convert to proper tensor
            waveform = _to_waveform_tensor(sample["array"], device)
            res = pipeline({"sample_rate": int(sample["sampling_rate"]),
                            "waveform": waveform})
            
            mask = np.full(res["snr"].shape, False)
            for (segment, _) in res["annotation"].itertracks():
                start = int(segment.start * ratio)
                end = int(segment.end * ratio)
                mask[start:end] = True
            mask =  (~((res["snr"] == 0.0) & (res["c50"] == 0.0)) & mask)

            vad_duration = sum(map(lambda x: x[0].duration, res["annotation"].itertracks()))
            
            snr.append(res["snr"][mask].mean())
            c50.append(res["c50"][mask].mean())
            vad_durations.append(np.float32(vad_duration))
        
        # 16ms window
        batch["snr"] = snr
        batch["c50"] = c50
        batch["speech_duration"] = vad_durations
        
    else:
        single = _ensure_sample_dict(val)
        waveform = _to_waveform_tensor(single["array"], device)
        res = pipeline({"sample_rate": int(single["sampling_rate"]),
                        "waveform": waveform})

        mask = np.full(res["snr"].shape, False)
        for (segment, _) in res["annotation"].itertracks():
            start = int(segment.start * ratio)
            end = int(segment.end * ratio)
            mask[start:end] = True
        mask =  (~((res["snr"] == 0.0) & (res["c50"] == 0.0)) & mask)

        vad_duration = sum(map(lambda x: x[0].duration, res["annotation"].itertracks()))     

        batch["snr"] = res["snr"][mask].mean()
        batch["c50"] = res["c50"][mask].mean()
        batch["speech_duration"] = vad_duration
        
    return batch
