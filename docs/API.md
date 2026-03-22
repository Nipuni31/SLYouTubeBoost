# HTTP API

## `POST /api/predict`

**Content-Type:** `application/json`

Body: one object with **all** feature keys expected by the current model (see `outputs/model_config.json` → `features`).

Example (v2 feature names):

```json
{
  "subscriber_count": 100000,
  "video_count": 200,
  "years_active": 3.0,
  "avg_views_per_video": 50000,
  "views_last_30_days": 1000000,
  "upload_frequency": 5.5,
  "engagement_ratio": 2.5,
  "upload_consistency": 1.2
}
```

**Success (200):**

```json
{
  "ok": true,
  "prediction": 0,
  "probability": 0.42,
  "summary": "...",
  "xai_strengths": [],
  "xai_weaknesses": [],
  "human_insights": ["..."],
  "shap_available": true
}
```

**Validation error (400):**

```json
{
  "ok": false,
  "errors": ["subscriber_count: must be non-negative"]
}
```

**Rules:** every value must be a **non-negative** number and not above `1e15` (see `MAX_FEATURE_VALUE` in `app.py`).

**curl:**

```bash
curl -s -X POST http://127.0.0.1:5000/api/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"subscriber_count\":1000,\"video_count\":50,...}"
```

HTML UI remains at `GET /` and `POST /predict`.
