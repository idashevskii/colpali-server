
## Local Development

Copy file `.env.example` to `.env` and adjust configuration.

Run:

```bash
docker compose up -d --build
```

Open in browser:
- [API](http://localhost:9001/)
- [API Docs](http://localhost:9001/docs)


### OPTIONAL: Install dependecies on host (for dev only)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
deactivate
```


## Deploy to Production

Copy file `.env.example` to `.env` and adjust configuration.

Run:

```bash
./scripts/prod-run.sh
```


## Acknowlegment
* [ColPali](https://github.com/illuin-tech/colpali): Efficient Document Retrieval with Vision Language Models (License MIT).
* [FastAPI](https://github.com/fastapi/fastapi): Web framework for building APIs with Python (License MIT).
