# MSDL-JCI

MSDL-JCI is a Python project for macro-economic feature engineering and future market-prediction workflows around JCI.  
The codebase has been refactored into a simple, FastAPI-style backend structure so it is easier to maintain, extend, and later connect to PostgreSQL or other services.

## Goals of the current structure

- Keep business logic separated from data access
- Make the code easier to test and replace piece by piece
- Prepare the project for an eventual FastAPI API layer
- Keep the current CSV-based workflow working for now

## Project Structure

```text
src/msdl_jci/
|-- core/
|   |-- config.py        # application settings, environment loading
|   `-- logging.py       # centralized logging setup
|-- domain/
|   `-- macro/
|       `-- encoder.py   # macro feature encoding business logic
|-- services/
|   `-- macro_service.py # use-case orchestration layer
|-- infrastructure/
|   `-- data_sources/
|       `-- csv_reader.py # CSV input adapter
`-- main.py              # current entrypoint
```

### Folder responsibilities

- `core`: shared app-level utilities like config and logging
- `domain`: pure business logic, with no direct file or database coupling
- `services`: application use cases that coordinate domain logic
- `infrastructure`: external adapters such as CSV, database, HTTP, or filesystem access
- `main.py`: current runnable entrypoint for local execution

## Why this structure

This layout follows a common backend pattern used in FastAPI projects:

- the `domain` layer stays stable even if the data source changes
- the `infrastructure` layer can later be swapped from CSV to PostgreSQL
- the `services` layer keeps workflows readable and easy to extend
- the `core` layer centralizes shared configuration and logging

## Current Workflow

Right now the main pipeline:

1. loads macro-economic CSV data
2. converts values into numeric series
3. trains MLP-based encoders
4. combines the encoded macro features into one array

## How to run

Use the virtual environment interpreter:

```powershell
.\.venv\Scripts\python.exe -m msdl_jci.main
```

Or, if you install the package entrypoint later, you can run the console script:

```powershell
msdl-jci
```

## Development notes

- The current data source is still CSV-based
- PostgreSQL support is planned for a later step
- The refactor intentionally avoids mixing database logic into domain code
- Legacy one-off scripts were removed to keep the repo focused and maintainable

## Future FastAPI direction

When the API layer is added, the project can grow into something like:

```text
src/msdl_jci/
|-- api/
|   |-- routes/
|   |-- schemas/
|   `-- dependencies.py
|-- core/
|-- domain/
|-- services/
`-- infrastructure/
```

That would keep HTTP concerns separate from the actual feature engineering and database access.
