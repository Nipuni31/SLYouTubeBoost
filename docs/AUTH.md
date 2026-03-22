# Authentication (JWT)

The app uses **Flask-JWT-Extended** with:

- **SQLite** user store (`instance/users.db`) — see `auth/db.py`
- **Password hashing** — `pbkdf2:sha256` via Werkzeug
- **Access + refresh tokens** — access ~1h, refresh ~30 days (see `app.py` config)

## Environment

| Variable | Description |
|----------|-------------|
| `JWT_SECRET_KEY` | Secret used to sign JWTs (set in production; defaults to `FLASK_SECRET_KEY` if unset) |
| `FLASK_SECRET_KEY` | Flask session secret |
| `DEFAULT_ADMIN_USER` / `DEFAULT_ADMIN_PASSWORD` | Seed user when the DB has no users |
| `AUTH_DISABLED` | If `true` / `1` / `yes`, JWT is not required (dev only) |
| `JWT_COOKIE_SECURE` | If `true`, cookies are `Secure` (use with HTTPS) |

## Web UI

1. Open `/` — if not logged in, you are redirected to `/auth/login`.
2. **Register** at `/auth/register` or sign in at `/auth/login`.
3. After login, **access** and **refresh** tokens are stored in **HTTP-only cookies** (`access_token_cookie`, `refresh_token_cookie`).
4. **Log out** via `POST /auth/logout` (forms in the UI).

## API clients (`/api/predict`)

Send the access token in either way:

1. **Header (recommended):**  
   `Authorization: Bearer <access_token>`

2. **Cookies:** same as the browser after logging in via `/auth/login` or `/auth/register`.

### Obtain tokens (JSON)

- `POST /auth/token` or `POST /auth/login/json`  
  Body: `{"username": "...", "password": "..."}`  
  Response: `access_token`, `refresh_token`, `expires_in`, etc.

### Refresh access token

- `POST /auth/refresh`  
  Header: `Authorization: Bearer <refresh_token>`  
  Response: new `access_token` (and cookie update if using cookies).

### Current user

- `GET /auth/me` with `Authorization: Bearer <access_token>`

## Security notes

- Change `DEFAULT_ADMIN_PASSWORD` and `JWT_SECRET_KEY` before production.
- Use `JWT_COOKIE_SECURE=true` with HTTPS.
- `AUTH_DISABLED` must stay **false** in production.
