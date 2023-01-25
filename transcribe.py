import glob
import json
import os
import random
import tempfile
import time
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor as TPE
from queue import Queue
from pydub.effects import normalize

import requests
from natsort import natsorted
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm

import com.amivoice.hrp.Hrp
import com.amivoice.hrp.HrpListener

__all__ = ["_transcribe"]


def _split(wav_path, output_dir):
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
    api_key = os.getenv("API_KEY")
    engine_name = os.getenv("ENGINE_NAME")
    endpoint = os.getenv("ENDPOINT_URL")
    codec = "16K"
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
            api = AsyncHrp(endpoint, codec, engine_name, api_key)
            tmp_results = dict()
            for f in files:
                res: Future = api(f)
                tmp_results[f] = res

            for f, r in tqdm(tmp_results.items(), total=len(files)):
                try:
                    tmp = r.result()
                    if tmp is not None:
                        results[f] = tmp
                    else:
                        org_files.remove(f)
                    files.remove(f)
                except Exception as e:
                    print(e)

        data = list()
        for f in org_files:
            res = results[f]
            data.append(dict(duration_seconds=get_seconds(f), results=res))

        join_wavs(org_files, wav_path.replace(".wav", "_slim.wav"))
    return pretty(data)


def time2str(t):
    hours, remainder = divmod(t, 3600 * 1000)
    minutes, remainder = divmod(remainder, 60 * 1000)
    seconds, miliseconds = divmod(remainder, 1000)
    return "{:02}:{:02}:{:02}.{:02}".format(
        int(hours), int(minutes), int(seconds), int(miliseconds)
    )


def append_new_lines(cands, len_thre=100):
    if len(cands) == 0:
        return ""
    lines = list()
    t = cands[0]["t"]  # time
    i = 0
    tmp_len = 0
    last_speaker = None
    for j, cand in enumerate(cands):
        tmp_len += len(cand["written"])
        speaker = cand["speaker"]
        # 改行の判定
        if (
            tmp_len >= len_thre
            or (
                cand["written"] == "。"
                or (cand["written"] == "、" and tmp_len >= len_thre // 2)
            )
            or (last_speaker is not None and last_speaker != speaker)
        ):
            s = "".join([x["written"] for x in cands[i : j + 1]])
            line = f"{time2str(t)}: \t{s}"
            lines.append(line)
            last_speaker = speaker
            if j + 1 < len(cands):
                i = j + 1
                t = cand["t"]
                tmp_len = 0

    return "\n".join(lines) + "\n"


def pretty(results):
    text = ""
    t_base = 0  # miliseconds
    t = 0
    cands = list()
    for idx, rr in enumerate(results):
        for result in rr["results"]:
            for res in result["results"]:
                for token in res["tokens"]:
                    t = t_base + int(token["starttime"])
                    speaker = token.get("label", "speaker")
                    cands.append(
                        {
                            "t": t,  # time
                            "written": token["written"],
                            "spoken": token["spoken"],
                            "confidence": float(token["confidence"]),
                            "speaker": f"{speaker}-{idx}",
                        }
                    )
                text += append_new_lines(cands)
                cands.clear()
        t_base += rr["duration_seconds"] * 1000
    
    text = text.strip()
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text