-- Discovery: photos hosted elsewhere (used by the dummy profiles) and sparks.

-- A media row either points at a file under MEDIA_DIR (storage_path) or at a public
-- image elsewhere (external_url); /api/v1/media/{id} redirects to the latter.
ALTER TABLE media ADD COLUMN IF NOT EXISTS external_url text;

-- One row each time someone sends a spark to another person (Send Spark on the deck).
-- A pass is not recorded.
CREATE TABLE IF NOT EXISTS sparks (
    from_user   uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    to_user     uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (from_user, to_user),
    CHECK (from_user <> to_user)
);

CREATE INDEX IF NOT EXISTS sparks_to_user_idx ON sparks (to_user);
