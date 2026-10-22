-- One profile per account, and the photos/voice files it uses.

CREATE TABLE IF NOT EXISTS media (
    -- Random and unguessable: the id doubles as the file's public link.
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    kind          text NOT NULL CHECK (kind IN ('photo', 'voice')),
    content_type  text NOT NULL,
    byte_size     integer NOT NULL CHECK (byte_size > 0),
    -- Path under MEDIA_DIR.
    storage_path  text NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS media_user_idx ON media (user_id);

CREATE TABLE IF NOT EXISTS profiles (
    user_id                 uuid PRIMARY KEY REFERENCES users (id) ON DELETE CASCADE,
    display_name            text NOT NULL,
    age                     smallint CHECK (age BETWEEN 18 AND 99),
    neighborhood            text NOT NULL DEFAULT '',
    craft                   text NOT NULL DEFAULT '',
    bio                     text NOT NULL DEFAULT '',
    intent                  text NOT NULL DEFAULT 'Lasting intimacy',
    voice_title             text NOT NULL DEFAULT '',
    voice_media_id          uuid REFERENCES media (id) ON DELETE SET NULL,
    voice_duration_seconds  real NOT NULL DEFAULT 0,
    -- Exactly three slots (cover, photo 2, photo 3), each a media id or null.
    photo_media_ids         jsonb NOT NULL DEFAULT '[null, null, null]',
    -- Up to two {title, body} vignettes.
    vignettes               jsonb NOT NULL DEFAULT '[]',
    -- What matches can see.
    visibility              jsonb NOT NULL DEFAULT '{"vibe_deck_highlights": true, "value_pillars": true, "call_reflections": false}',
    created_at              timestamptz NOT NULL DEFAULT now(),
    updated_at              timestamptz NOT NULL DEFAULT now()
);
