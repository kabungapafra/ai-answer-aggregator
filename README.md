# AI Answer Aggregator

A Python program that queries multiple AI models simultaneously and combines their responses into a single, coherent "super answer" by summarizing and resolving contradictions.

## Features

- **Multi-Model Querying**: Sends questions to 5 AI models in parallel for efficiency
- **Intelligent Aggregation**: Combines answers by identifying common themes and resolving contradictions
- **Error Handling**: Gracefully handles API errors and missing responses
- **Modular Design**: Easy to extend with additional AI models
- **Parallel Processing**: Uses threading for fast concurrent API calls

## Currently Supported Models

1. OpenAI GPT-4
2. OpenAI GPT-3.5 Turbo
3. Anthropic Claude 3 Sonnet
4. Google Gemini Pro
5. OpenAI GPT-4 Turbo

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Set the following environment variables with your API keys:

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your-openai-api-key"
$env:ANTHROPIC_API_KEY="your-anthropic-api-key"
$env:GOOGLE_API_KEY="your-google-api-key"
```

**Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=your-openai-api-key
set ANTHROPIC_API_KEY=your-anthropic-api-key
set GOOGLE_API_KEY=your-google-api-key
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GOOGLE_API_KEY="your-google-api-key"
```

Alternatively, you can create a `.env` file (requires `python-dotenv` package) or modify the code to load keys from a config file.

### 3. Run the Program

```bash
python ai_aggregator.py
```

## Usage

1. Run the program
2. Enter your question when prompted
3. The program will:
   - Query all 5 AI models in parallel
   - Display individual responses
   - Synthesize a final coherent answer
   - Display the aggregated result

## How It Works

1. **Query Phase**: The program sends your question to all configured AI models simultaneously using parallel threading
2. **Collection Phase**: Responses (and any errors) are collected from each model
3. **Aggregation Phase**: A synthesis AI model combines all successful responses, identifying:
   - Common themes and agreements
   - Contradictions and resolving them
   - Best insights from each response
4. **Output Phase**: The final synthesized answer is displayed

## Adding More AI Models

To add a new AI model, simply add a new entry to the `AI_MODELS` dictionary in `ai_aggregator.py`:

```python
AI_MODELS = {
    # ... existing models ...
    "new_model_name": {
        "provider": "provider_name",  # Must match a provider handler
        "model": "model-identifier",
        "api_key_env": "API_KEY_ENV_VAR"
    }
}
```

Then implement the query function `_query_provider_name()` if the provider doesn't exist yet.

## Error Handling

The program handles various error scenarios:
- Missing API keys
- Network errors
- API rate limits
- Invalid responses
- Provider-specific errors

If a model fails, the program continues with the remaining models and notes which ones failed.

## Requirements

- Python 3.8+
- Valid API keys for at least one of the supported AI providers
- Internet connection

## License

This project is provided as-is for educational and personal use.

