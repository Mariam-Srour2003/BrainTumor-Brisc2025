Simple frontend to interact with the ML service API.

Run the API locally (from repo root):

```bash
# ensure FastAPI and uvicorn are installed in your environment
python -m pip install fastapi uvicorn
# run the API
uvicorn ml_service.api.main:create_app --factory --reload
```

Then serve the static frontend (or open file directly). Quick server:

```bash
python -m http.server --directory frontend/simple 8001
# open http://localhost:8001 in your browser
```

Notes:
- The frontend assumes the API is at http://localhost:8000. Edit the API base URL in the UI if different.
- Training operations can be long; the UI triggers endpoints synchronously and will show returned JSON once complete.
