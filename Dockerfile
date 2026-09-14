FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs

RUN pip install --upgrade pip && pip install .

CMD ["msdl-jci", "--help"]
