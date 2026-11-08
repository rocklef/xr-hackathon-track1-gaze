"""Generate the voiceover (edge-tts neural voice) + word timings for subtitles.

    python video/narration.py                     # default voice
    python video/narration.py --voice en-IN-PrabhatNeural

Writes video/build/voice/<scene>.mp3 and <scene>.json (word timings, seconds).
"""
import argparse
import asyncio
import json
import os

import edge_tts

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "video", "build", "voice")

SCRIPT = {
    "title": "Hi, we're Team Rocklef, and this is our Track One project.",
    "background": "Imagine a headset that sharpens only what you look at, a car that notices a driver's drifting eyes, "
                  "or a screen you control with just your eyes. Each one needs to know exactly where you're looking.",
    "problem": "But these systems learn from examples, and gathering real ones is slow, costly and private, and never "
               "perfectly labelled. So they fail where they've seen the least.",
    "literature": "Researchers have tried two routes: recording thousands of real volunteers, and drawing eyes with "
                  "computer graphics. Both work, but each trades off label accuracy, realism, or control.",
    "solution": "The gap: no open tool where both eyes truly fixate a 3D point, through a fixed headset-style camera, "
                "with labels you can verify. So we built one.",
    "architecture": "We build a textured face, find each eyeball's true centre and axis, and rig both eyes onto a moving "
                    "target, with eyelids that follow. A locked camera renders each frame with full ground truth, "
                    "and a verifier checks it.",
    "setup": "The fixation target sits thirty-five centimetres ahead and moves up to five centimetres either way. The "
             "camera is locked forty centimetres ahead, and both lines of sight meet on the target.",
    "gaze": "In Python, eyes dot move, with h and v, turns both eyes together, and eyes dot reset re-centres them. "
            "Notice the eyelids following the gaze.",
    "wrapper": "One command renders the ten clockwise frames from the top, with metadata for each: the inputs, eye "
               "rotations, two D pupil centres and the full camera matrix.",
    "proof": "An independent checker then finds the pupils in the pixels themselves. All five directions pass, and all "
             "ten frames follow the clockwise order.",
    "accuracy": "During calibration, we paint the pupil green and fit its true position behind the cornea, so our labels "
                "are accurate to under half a pixel.",
    "compare": "Against the state of the art: real datasets are costly, private and approximate. Graphics renderers are "
               "fast, but mostly model one eye. Ours is free, private, binocular, and proves every label from the pixels.",
    "novelty": "What's new: pupils calibrated through the cornea, true binocular vergence, gaze-linked eyelids, and a "
               "self-checking pipeline. And it reaches far beyond headsets: driver monitoring, eye-controlled "
               "accessibility, medical eye screening, and digital humans.",
    "future": "Next: infrared headset cameras, more faces, skin tones and glasses, natural blinks, and testing how well "
              "models trained on our images work on real eyes.",
    "outro": "Calibrated, verifiable eye data from one script. For the code or further development, email us at the "
             "addresses on screen. Thank you, from Team Rocklef.",
}


async def synth(key, text, voice, rate):
    com = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
    words, audio = [], bytearray()
    async for ch in com.stream():
        if ch["type"] == "audio":
            audio.extend(ch["data"])
        elif ch["type"] == "WordBoundary":
            words.append({"w": ch["text"], "t0": ch["offset"] / 1e7, "t1": (ch["offset"] + ch["duration"]) / 1e7})
    with open(os.path.join(OUT, f"{key}.mp3"), "wb") as f:
        f.write(audio)
    with open(os.path.join(OUT, f"{key}.json"), "w") as f:
        json.dump({"text": text, "words": words}, f, indent=1)
    print(f"  {key:9s} {words[-1]['t1']:5.1f}s  {len(words)} words")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="en-US-AndrewNeural")
    ap.add_argument("--rate", default="+6%")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for k, t in SCRIPT.items():
        for attempt in range(5):
            try:
                await synth(k, t, a.voice, a.rate)
                break
            except edge_tts.exceptions.NoAudioReceived:
                print(f"  {k}: no audio, retry {attempt + 1}")
                await asyncio.sleep(2 + 2 * attempt)
        else:
            raise SystemExit(f"edge-tts failed for {k}")
    with open(os.path.join(OUT, "voice.txt"), "w") as f:
        f.write(a.voice)


if __name__ == "__main__":
    asyncio.run(main())
