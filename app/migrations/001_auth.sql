-- Accounts and login sessions.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS users (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    email           text NOT NULL,
    display_name    text NOT NULL,
    -- NULL for accounts that only use Google sign-in.
    password_hash   text,
    auth_provider   text NOT NULL DEFAULT 'password' CHECK (auth_provider IN ('password', 'google')),
    -- Google's stable account id, filled when Google sign-in is integrated.
    google_sub      text UNIQUE,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),
    last_login_at   timestamptz
);

-- Emails are compared case-insensitively.
CREATE UNIQUE INDEX IF NOT EXISTS users_email_lower_key ON users (lower(email));

-- One row per signed-in device. Only a hash of the refresh token is stored.
CREATE TABLE IF NOT EXISTS auth_sessions (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    refresh_token_hash  text NOT NULL UNIQUE,
    device_label        text,
    created_at          timestamptz NOT NULL DEFAULT now(),
    last_used_at        timestamptz NOT NULL DEFAULT now(),
    expires_at          timestamptz NOT NULL,
    revoked_at          timestamptz
);

CREATE INDEX IF NOT EXISTS auth_sessions_user_idx ON auth_sessions (user_id);
