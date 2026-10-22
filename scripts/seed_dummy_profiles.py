"""Create 10 dummy accounts with finished profiles for the swipe deck.

Run from backend/:   .venv/bin/python scripts/seed_dummy_profiles.py

- Safe to run again: existing dummy accounts (emails ending @dummy.synq.test) are
  deleted first, together with their profiles, media rows, sparks and voice files.
- Dummy accounts have no password, so nobody can sign in as them.
- Photos are public stock photos (Unsplash), stored as media rows with an
  external_url; /api/v1/media/{id} redirects to them.
- Each voice reflection is spoken by the laptop's Piper voice (en_US-amy-medium) and
  converted to M4A with ffmpeg. There is only one (female) voice, so the men's
  recordings are pitched down a little; they are clearly synthetic test audio.

Test data only: these people don't exist.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import asyncpg

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from app import db  # noqa: E402
from app.config import settings  # noqa: E402

DOMAIN = "dummy.synq.test"
PIPER = BACKEND / ".venv" / "bin" / "piper"
PIPER_MODEL = BACKEND / "en_US-amy-medium.onnx"


def unsplash(photo_id: str) -> str:
    return f"https://images.unsplash.com/photo-{photo_id}?auto=format&fit=crop&w=1000&q=85"


@dataclass
class Person:
    slug: str
    name: str
    age: int
    male: bool
    neighborhood: str
    craft: str
    bio: str
    intent: str
    voice_title: str
    voice_text: str
    vignettes: list[tuple[str, str]]
    photos: list[str]  # cover, photo 2, photo 3 (Unsplash ids)


PEOPLE = [
    Person(
        "mira", "Mira", 28, False, "Juhu, Mumbai", "Documentary filmmaker",
        "I chase small, true stories with a camera and too much chai. Happiest on a rainy local train, "
        "notebook open, eavesdropping for dialogue. Looking for someone curious enough to get lost with me.",
        "Long-term partner", "A city after rain",
        "A city feels completely different after rain. The noise softens, people slow down, and every window looks like a story.",
        [("Rain-soaked Sundays", "Monsoon walks along Juhu with cutting chai and no plan at all."),
         ("Frames I keep", "I still shoot one roll of film a month. Patience is the whole point.")],
        ["1524504388940-b1c1722653e1", "1428592953211-077101b2021b", "1502920917128-1aa500764cbd"],
    ),
    Person(
        "kabir", "Kabir", 30, True, "Bandra West, Mumbai", "Architect & writer",
        "I design homes for a living and write about the ones we carry inside us. Slow mornings, long letters, "
        "old buildings with good light. I'd love a partner who likes deep talks and quiet company equally.",
        "Lasting intimacy", "Why slow mornings matter",
        "Slow mornings matter to me. Coffee before the phone, sunlight on the floor, and one honest conversation before the day begins.",
        [("Sunday ritual", "Pour-over coffee, the newspaper, and sketching whatever the light is doing."),
         ("Shelf of favourites", "Too many books, all with pencil notes in the margins.")],
        ["1500648767791-00dcc994a43e", "1495474472287-4d71bcdd2085", "1521587760476-6c12a4b040da"],
    ),
    Person(
        "ananya", "Ananya", 27, False, "Colaba, Mumbai", "Pastry chef",
        "I bake for a living and feed people as a love language. Early riser, late-night dessert believer, "
        "terrible at small talk but great at second helpings. Kindness is my favourite flavour.",
        "Long-term partner", "The smell of butter at six a.m.",
        "There is nothing like the smell of butter at six in the morning. The kitchen is quiet, and everything still feels possible.",
        [("Market mornings", "Crawford Market at dawn for the best fruit before the city wakes up."),
         ("Plates I love", "Simple food, done properly. Fresh herbs make everything better.")],
        ["1494790108377-be9c29b29330", "1504674900247-0877df9cc836", "1571934811356-5cc061b6821f"],
    ),
    Person(
        "ayan", "Ayan", 31, True, "Pali Hill, Mumbai", "Chef & music collector",
        "Chef by day, vinyl hunter by weekend. I believe the best dinners never need a plan and the best "
        "songs are on side B. Looking for someone who'll dance badly with me in the kitchen.",
        "Open to see", "Side B songs",
        "The best dinners never need a plan. Put on an old record, open the fridge, and see what kind of evening it wants to be.",
        [("Guitar evenings", "I play badly, but with feeling. Requests are welcome."),
         ("Weekend escapes", "Lonavala hills, misty drives, and a thermos of masala chai.")],
        ["1506794778202-cad84cf45f1d", "1510915361894-db8b60106cb1", "1464822759023-fed622ff2c3b"],
    ),
    Person(
        "rohan", "Rohan", 29, True, "Powai, Mumbai", "Product designer",
        "I make apps feel calm. Off screen I cycle around the lake at dawn and keep far too many plants. "
        "Into honest conversations, good design, and people who laugh at their own jokes.",
        "Long-term partner", "Dawn rides around the lake",
        "My favourite hour is just before sunrise, cycling around the lake while the whole city is still asleep.",
        [("Morning miles", "Twenty kilometres before breakfast, then the best idli in Powai."),
         ("Plant parent", "Thirty-one plants and counting. They all have names.")],
        ["1507003211169-0a1dd7228f2d", "1485965120184-e220f721d03e", "1416879595882-3373a0480b5b"],
    ),
    Person(
        "ira", "Ira", 26, False, "Versova, Mumbai", "Ceramic artist",
        "I make bowls that are slightly uneven on purpose. Sea air, clay under my nails, and long dinners "
        "with friends. I fall for people who notice small things.",
        "Lasting intimacy", "Imperfect on purpose",
        "I like things that are a little imperfect. A bowl with a thumbprint, a laugh that comes out too loud. That is where the warmth is.",
        [("Studio days", "Wheel, glaze, kiln, repeat. Every piece surprises me."),
         ("Sea at sunset", "Versova jetty at golden hour is my reset button.")],
        ["1534528741775-53994a69daeb", "1513364776144-60967b0f800f", "1495616811223-4d98c6e9c869"],
    ),
    Person(
        "kiara", "Kiara", 29, False, "Lower Parel, Mumbai", "Brand strategist",
        "I help brands find their voice and still struggle to choose a dinner spot. Beach weekends, "
        "spontaneous road trips, and playlists for every mood. Let's skip the small talk.",
        "Open to see", "Saltwater weekends",
        "Give me a quiet beach, a good book, and no network signal for a weekend. That is my idea of luxury.",
        [("Beach therapy", "Alibaug ferry on a Friday evening fixes almost everything."),
         ("City nights", "Rooftops, skyline views, and conversations that run past midnight.")],
        ["1544005313-94ddf0286df2", "1507525428034-b723cf961d3e", "1477959858617-67f85cf4f1df"],
    ),
    Person(
        "arjun", "Arjun", 27, True, "Andheri West, Mumbai", "Indie musician",
        "I write songs about trains, rain and people I haven't met yet. Open mics on weeknights, temple "
        "tea on weekends. Looking for someone who'd sit through a soundcheck and still smile.",
        "Lasting intimacy", "Songs about trains",
        "Most of my songs start on a local train. Everyone is going somewhere, and every face has a chorus hiding in it.",
        [("Local train notebook", "Lyrics scribbled between Andheri and Churchgate."),
         ("Gig nights", "Small rooms, warm lights, and strangers singing along.")],
        ["1539571696357-5a69c17a67c6", "1474487548417-781cb71495f3", "1510915361894-db8b60106cb1"],
    ),
    Person(
        "naina", "Naina", 30, False, "Dadar, Mumbai", "Paediatric doctor",
        "I spend my days making tiny humans laugh through their vaccines. Weekends are for running, my "
        "beagle, and cooking my mother's recipes badly. Steady, warm, and a little goofy.",
        "Long-term partner", "Small brave faces",
        "The bravest people I know are five years old. They cry, they hold your hand, and then they ask for a sticker.",
        [("Shivaji Park runs", "Early morning laps before the cricket crowd arrives."),
         ("Meet Biscuit", "My beagle, who believes every stranger is a new best friend.")],
        ["1531123897727-8f129e1688ce", "1476480862126-209bfaa8edc8", "1543466835-00a7907e9de1"],
    ),
    Person(
        "vihaan", "Vihaan", 32, True, "Worli, Mumbai", "Investment analyst",
        "Spreadsheets by day, sea-face walks by night. I read history, cook on Sundays, and plan trips I "
        "actually take. Hoping to meet someone who wants something real and unhurried.",
        "Lasting intimacy", "The sea face at night",
        "Walking along the sea face at night clears my head. The waves keep the same rhythm no matter how loud the day was.",
        [("Sunday kitchen", "Slow-cooked dal and a playlist of old Hindi songs."),
         ("Next trip", "Ladakh by road. The plan is finished; the courage is pending.")],
        ["1519085360753-af0119f7cbe7", "1505142468610-359e7d316be0", "1464822759023-fed622ff2c3b"],
    ),
]


def make_voice(text: str, male: bool, out_path: Path) -> float:
    """Speak `text` with Piper and save it as M4A. Returns the length in seconds."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "voice.wav"
        subprocess.run([str(PIPER), "-m", str(PIPER_MODEL), "-f", str(wav)], input=text.encode(), check=True, capture_output=True)
        # Men: lower the pitch by ~15% while keeping the speed.
        audio_filter = "asetrate=22050*0.85,aresample=22050,atempo=1.176" if male else "anull"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-af", audio_filter, "-c:a", "aac", "-b:a", "96k", str(out_path)],
            check=True,
        )
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(out_path)],
        check=True, capture_output=True, text=True,
    )
    return round(float(probe.stdout.strip()), 1)


async def seed() -> None:
    # Same limits as the profile API (ProfileIn), since this script writes to the table directly.
    for person in PEOPLE:
        assert len(person.bio) <= 280 and len(person.voice_title) <= 70 and len(person.craft) <= 80, person.name
        assert len(person.neighborhood) <= 120 and 18 <= person.age <= 99 and len(person.vignettes) <= 2, person.name
    if not settings.database_url:
        raise SystemExit("DATABASE_URL is not set in backend/.env")
    if not PIPER.exists() or not PIPER_MODEL.exists() or shutil.which("ffmpeg") is None:
        raise SystemExit("Piper (with en_US-amy-medium.onnx) and ffmpeg are needed for the voice reflections")
    pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=2)
    await db.migrate(pool)
    async with pool.acquire() as connection:
        old = await connection.fetch("SELECT id FROM users WHERE email LIKE $1", f"%@{DOMAIN}")
        for row in old:
            shutil.rmtree(settings.media_dir / str(row["id"]), ignore_errors=True)
        await connection.execute("DELETE FROM users WHERE email LIKE $1", f"%@{DOMAIN}")
        for person in PEOPLE:
            async with connection.transaction():
                user_id = await connection.fetchval(
                    "INSERT INTO users (email, display_name, password_hash) VALUES ($1, $2, NULL) RETURNING id",
                    f"{person.slug}@{DOMAIN}", person.name,
                )
                photo_ids = []
                for photo in person.photos:
                    media_id = uuid4()
                    await connection.execute(
                        "INSERT INTO media (id, user_id, kind, content_type, byte_size, storage_path, external_url) "
                        "VALUES ($1, $2, 'photo', 'image/jpeg', 1, '', $3)",
                        media_id, user_id, unsplash(photo),
                    )
                    photo_ids.append(str(media_id))
                voice_id = uuid4()
                relative = f"{user_id}/{voice_id}.m4a"
                seconds = make_voice(person.voice_text, person.male, settings.media_dir / relative)
                size = (settings.media_dir / relative).stat().st_size
                await connection.execute(
                    "INSERT INTO media (id, user_id, kind, content_type, byte_size, storage_path) VALUES ($1, $2, 'voice', 'audio/mp4', $3, $4)",
                    voice_id, user_id, size, relative,
                )
                await connection.execute(
                    """
                    INSERT INTO profiles (user_id, display_name, age, neighborhood, craft, bio, intent, voice_title,
                                          voice_media_id, voice_duration_seconds, photo_media_ids, vignettes, visibility)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, $12::jsonb, $13::jsonb)
                    """,
                    user_id, person.name, person.age, person.neighborhood, person.craft, person.bio, person.intent,
                    person.voice_title, voice_id, min(seconds, 30.0), json.dumps(photo_ids),
                    json.dumps([{"title": title, "body": body} for title, body in person.vignettes]),
                    json.dumps({"vibe_deck_highlights": True, "value_pillars": True, "call_reflections": False}),
                )
            print(f"  {person.name:7s} {person.age}  {person.neighborhood:22s} voice {seconds:4.1f}s")
    await pool.close()
    print(f"Seeded {len(PEOPLE)} dummy profiles.")


if __name__ == "__main__":
    asyncio.run(seed())
