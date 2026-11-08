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
    "title": "Hi, we're Team Rocklef. This is our Track One project: synthetic eye movement and virtual camera rendering.",
    "problem": "Eye trackers in X R headsets learn from thousands of labelled eye images. But capturing real eyes at "
               "extreme angles is slow, expensive, and restricted by privacy rules. So we built a calibrated synthetic "
               "eye rig in Blender, where every image comes with perfect labels.",
    "setup": "Here's the setup. A fixation target sits thirty-five centimetres in front of the eyes, and can move up to "
             "five centimetres in any direction. The render camera is locked forty centimetres ahead. Each eyeball "
             "rotates about its own true centre, so both lines of sight meet exactly on the target.",
    "gaze": "Our Python wrapper drives everything. Calling eyes dot move, with h and v, turns both eyes together: left, "
            "right, up and down. And eyes dot reset brings them back to centre. Notice how the eyelids follow the "
            "gaze, just like real eyes.",
    "wrapper": "A single command renders the ten clockwise frames, starting from the top. For every frame, it also "
               "writes the metadata: the inputs, both eye rotations, the two D pupil centres, and the full camera matrix.",
    "proof": "We don't just trust our own numbers. An independent checker finds the pupils directly in the rendered "
             "pixels. All five directions pass, and all ten frames follow the clockwise order.",
    "accuracy": "To make the labels precise, we paint the pupil green during calibration, and fit its true position "
                "behind the cornea. Our pupil labels are accurate to under half a pixel.",
    "outro": "Everything is reproducible from a single script. Thank you, from Team Rocklef.",
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
    ap.add_argument("--rate", default="+0%")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for k, t in SCRIPT.items():
        await synth(k, t, a.voice, a.rate)
    with open(os.path.join(OUT, "voice.txt"), "w") as f:
        f.write(a.voice)


if __name__ == "__main__":
    asyncio.run(main())
