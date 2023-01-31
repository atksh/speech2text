import glob
import os
import tempfile

import noisereduce as nr
import whisper
from natsort import natsorted
from pydub import AudioSegment
from pydub.effects import normalize
from pydub.silence import split_on_silence
from scipy.io import wavfile
from tqdm import tqdm


class Whisper:
    def __init__(self, size: str = "base") -> None:
        self.model = whisper.load_model(size)

    def __call__(self, path: str) -> str:
        result = self.model.transcribe(path, fp16=False)
        segments = result["segments"]
        texts = []
        for seg in segments:
            texts.append(seg["text"])
        text = "\n".join(texts)
        print(text)
        return text


def reduce_noise(path: str) -> None:
    print("reducing noise")
    rate, data = wavfile.read(path)
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    wavfile.write(path, rate, reduced_noise)


def _split(wav_path, output_dir):
    reduce_noise(wav_path)
    sound = AudioSegment.from_file(wav_path, format="wav")
    print("splitting on sience")
    chunks = split_on_silence(
        sound, min_silence_len=1000, silence_thresh=-80, keep_silence=200, seek_step=50
    )
    print("done splitting. Now exporting")
    for i, chunk in tqdm(enumerate(chunks)):
        out_path = os.path.join(output_dir, f"{i}.wav")
        chunk.export(out_path, format="wav")


def get_seconds(f):
    audio = AudioSegment.from_file(f)
    return audio.duration_seconds


def join_wavs(wav_paths, output_path):
    sounds = [AudioSegment.from_file(f) for f in wav_paths]
    combined_wav = AudioSegment.empty()
    for sound in sounds:
        combined_wav += normalize(sound)
    combined_wav = normalize(combined_wav)
    combined_wav.export(output_path, format="wav")


def _transcribe(wav_path):
    size = os.getenv("MODEL_SIZE", "base")
    with tempfile.TemporaryDirectory() as dname:
        _split(wav_path, dname)

        results = dict()
        files = natsorted(glob.glob(f"{dname}/*.wav"))

        # normalize
        for f in files:
            s = AudioSegment.from_file(f)
            s = normalize(s)
            s.export(f, format="wav")

        org_files = files.copy()

        while len(files) > 0:
            model = Whisper(size)
            tmp_results = dict()
            for f in tqdm(files):
                res = model(f)
                tmp_results[f] = res

            for f, r in tqdm(tmp_results.items(), total=len(files)):
                if not r:
                    org_files.remove(f)
                files.remove(f)

        data = list()
        for f in org_files:
            data.append(tmp_results[f])

        join_wavs(org_files, wav_path.replace(".wav", "_slim.wav"))
    return "\n\n".join(data).strip()
