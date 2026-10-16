"""Small terminal harness for the 15-turn text milestone."""

import asyncio

from .conversation import call_store, generate_turn


async def main() -> None:
    language = input("Language (english/hinglish) [english]: ").strip().lower() or "english"
    if language not in ("english", "hinglish"):
        print("Please choose english or hinglish.")
        return
    consent = input("Start a temporary text conversation? (yes/no): ").strip().lower()
    if consent not in ("yes", "y"):
        print("No session was created.")
        return
    session = await call_store.create(language)  # type: ignore[arg-type]
    assistant, provider = await generate_turn(session)
    print(f"\nBondAI [{provider}]: {assistant}")
    for turn_number in range(1, 16):
        user_text = input(f"\nYou ({turn_number}/15, /end to stop): ").strip()
        if user_text.lower() == "/end":
            break
        if not user_text:
            print("Share a short answer, or type /end.")
            continue
        assistant, provider = await generate_turn(session, user_text)
        print(f"\nBondAI [{provider}]: {assistant}")
    call_store.remove(session.call_id)
    print("\nSession ended and its in-memory transcript was discarded.")


if __name__ == "__main__":
    asyncio.run(main())
