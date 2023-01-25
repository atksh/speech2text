import os

import invoke


@invoke.task
def build(c):
    c.run("docker build -t stt .")


@invoke.task
def convert_to_wav(c, input_path, output_path):
    print(input_path, output_path)
    c.run(
        f"ffmpeg -i {input_path} -vn -ac 1 -acodec pcm_s16le {output_path} -y", pty=True
    )


@invoke.task
def transcribe(c, wav_path, output_path):
    from transcribe import _transcribe

    text = _transcribe(wav_path)
    with open(output_path, "w") as f:
        f.write(text)


@invoke.task
def do_all(c, file_path):
    d = "".join(file_path.split("/")[:-1])
    fname = "".join(file_path.split("/")[-1].split(".")[:-1])
    wav_path = f"{d}/{fname}.wav"
    text_path = f"{d}/{fname}.txt"
    c.run(f"inv convert-to-wav {file_path} {wav_path}", pty=True)
    c.run(f"inv transcribe {wav_path} {text_path}", pty=True)


@invoke.task
def stt(c, file_path):
    if not os.path.exists(file_path):
        file_path = os.path.join("data", file_path)
    if not os.path.exists(file_path):
        assert FileNotFoundError

    gpu_available = os.getenv("GPU_AVAILABLE", "false") == "true"
    if gpu_available:
        gpu_arg = "--gpus all"
    c.run(
        (
            f"docker run {gpu_arg} -v $(pwd):/code --rm -it --env-file .env --entrypoint inv stt "
            f"do-all {file_path}"
        ),
        pty=True,
    )
