# Fano Platform

A platform of autonomous agents for exploring mathematical structures through the Fano plane.

## Components

### `explorer/`
The main exploration application. Uses multi-LLM consensus to discover, verify, and curate mathematical insights connecting Fano plane geometry to music theory, yoga, and cosmology.

```bash
cd explorer
python fano_explorer.py --help
```

### `pool/`
Browser Pool Service (coming soon). A shared HTTP service that manages browser instances for LLM access, providing:
- Unified authentication
- Rate limit tracking
- Request queueing with priorities

### `shared/`
Common utilities shared across components.

## Setup

1. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r explorer/requirements.txt
   ```

3. Configure API keys in `.env`:
   ```
   ANTHROPIC_API_KEY=your_key
   OPENROUTER_API_KEY=your_key
   ```

4. Authenticate browsers:
   ```bash
   cd explorer
   python fano_explorer.py auth
   ```

## Documentation

See `explorer/docs/` for detailed documentation.
