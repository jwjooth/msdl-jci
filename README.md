# MSDL-JCI

This project is a clean Python codebase for macro-economic feature engineering and JCI-related analysis. It is now arranged in a simple thesis-friendly structure so it is easy to read, maintain, and extend without forcing a backend-style architecture.

## Project Goals

- Keep the code simple and readable
- Separate reusable logic from script-style execution
- Make maintenance easier for long-term development
- Keep performance reasonable without overengineering
- Leave room for future database or API work if needed

## Current Folder Structure

```text
src/msdl_jci/
|-- config.py
|-- data_loader.py
|-- logging_config.py
|-- macro_encoder.py
|-- pipeline.py
`-- main.py
```

### What each file does

- `config.py`: project settings, paths, and environment loading
- `data_loader.py`: CSV reading and numeric cleaning helpers
- `logging_config.py`: reusable logging setup
- `macro_encoder.py`: core macro feature encoding logic
- `pipeline.py`: simple orchestration wrapper for the workflow
- `main.py`: runnable entrypoint

## Why this structure

This layout keeps the project practical for thesis development:

- no unnecessary backend layers
- business logic stays in one place
- data access is separated from model logic
- the workflow is easy to trace from top to bottom
- future changes will be easier to make because responsibilities are clear

## Workflow

The current pipeline:

1. reads macro-economic CSV data
2. cleans and converts numeric series
3. trains the MLP-based encoder
4. combines the encoded macro features into one matrix

## How to run

Use the virtual environment interpreter:

```powershell
.\.venv\Scripts\python.exe -m msdl_jci.main
```

## Best-Practice Notes

- Keep reusable logic inside modules, not inside one-off scripts
- Put file reading and preprocessing in separate helpers
- Keep the main entrypoint thin
- Avoid mixing database, file, and ML logic in the same file
- Prefer small, focused modules over deep folder nesting

## What was cleaned up

- removed the previous FastAPI-style layering
- removed duplicate legacy scripts and unused structure
- kept only the minimal modules needed for the current workflow

## Future Growth

If you later want to add more advanced features, you can still expand this structure gradually:

- add a `database/` module for PostgreSQL
- add a `features/` module if feature engineering grows
- add a `scripts/` folder for standalone experiments
- add tests for the loader, encoder, and pipeline

