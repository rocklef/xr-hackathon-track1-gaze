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
    "open": "Every second, your eyes show what you are paying attention to. And more and more machines are learning "
            "to read them.",
    "vr": "In a virtual reality headset, eye tracking means the picture only has to be perfectly sharp where you are "
          "looking. So it runs smoother, and the battery lasts longer.",
    "car": "In cars, it saves lives. Since July, every new car sold in Europe must warn a driver whose eyes leave "
           "the road.",
    "access": "For people who can't move or speak, like many living with ALS, the eyes become a keyboard. They can "
              "type, and even talk, just by looking.",
    "medical": "And for doctors, the way a young child's eyes follow a picture can help spot conditions like autism "
               "much earlier.",
    "title": "We're Team Rocklef, from SRM Institute of Science and Technology.",
    "problem": "But every one of these systems has to learn first, from huge numbers of eye photos, each labelled "
               "with exactly where the person was looking. Collecting them from real people is slow, expensive and "
               "private. And the labels are never perfect.",
    "literature": "Researchers have tried filming volunteers, and drawing eyes with computer graphics. But each "
                  "approach gives something up.",
    "solution": "So we built digital humans, with eyes we can point anywhere, and we always know the exact answer.",
    "eyeball": "Each eye turns around its true centre, and looks along its real line of sight, just like yours.",
    "setup": "A target in front of the face tells both eyes where to look, and a fixed camera takes the picture.",
    "gaze": "Left, right, up, down. Even the eyelids follow.",
    "architecture": "One script does it all: build the face, point the eyes, take the picture, save the answer, and "
                    "double check it.",
    "proof": "The computer then checks its own work, finding the pupils in every picture. All five directions pass. "
             "All ten frames, in the right order, accurate to under half a pixel.",
    "compare": "Compared with collecting real photos, it is faster, it protects privacy, it handles both eyes "
               "together, and it checks itself.",
    "future": "Next, we'll add more faces, infrared headset cameras and natural blinks, so the next headset, car or "
              "medical device can learn faster, and more safely.",
    "outro": "Want the code, or to build on it? Email us. Thank you, from Team Rocklef.",
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
