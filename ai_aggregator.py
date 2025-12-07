"""
AI Answer Aggregator
A program that queries multiple AI models and combines their responses into a single coherent answer.
"""

import os
import json
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional imports - handle missing packages gracefully
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic package not installed. Anthropic models will be skipped.")

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("Warning: google-generativeai package not installed. Google models will be skipped.")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai package not installed. OpenAI models will be skipped.")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests package not installed.")


# Configuration: Add your API keys here or use environment variables
# You can easily add more models by adding their configuration here
AI_MODELS = {
    "openai_gpt4": {
        "provider": "openai",
        "model": "gpt-4",
        "api_key_env": "OPENAI_API_KEY"
    },
    "openai_gpt35": {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "api_key_env": "OPENAI_API_KEY"
    },
    "anthropic_claude": {
        "provider": "anthropic",
        "model": "claude-3-sonnet-20240229",
        "api_key_env": "ANTHROPIC_API_KEY"
    },
    "google_gemini": {
        "provider": "google",
        "model": "gemini-pro",
        "api_key_env": "GOOGLE_API_KEY"
    },
    "openai_gpt4_turbo": {
        "provider": "openai",
        "model": "gpt-4-turbo-preview",
        "api_key_env": "OPENAI_API_KEY"
    }
}


def query_ai_model(model_name: str, model_config: Dict, question: str) -> Dict[str, any]:
    """
    Queries a single AI model with the user's question.
    
    Args:
        model_name: Name identifier for the model
        model_config: Configuration dictionary containing provider, model, and API key env var
        question: The user's question to send to the AI model
        
    Returns:
        Dictionary containing model_name, response (or None if error), and error message (if any)
    """
    provider = model_config["provider"]
    model = model_config["model"]
    api_key_env = model_config["api_key_env"]
    
    # Get API key from environment variable
    api_key = os.getenv(api_key_env)
    
    if not api_key:
        return {
            "model_name": model_name,
            "response": None,
            "error": f"API key not found for {model_name}. Please set {api_key_env} environment variable."
        }
    
    try:
        # Route to appropriate provider based on configuration
        if provider == "openai":
            return _query_openai(model, api_key, question, model_name)
        elif provider == "anthropic":
            return _query_anthropic(model, api_key, question, model_name)
        elif provider == "google":
            return _query_google(model, api_key, question, model_name)
        else:
            return {
                "model_name": model_name,
                "response": None,
                "error": f"Unknown provider: {provider}"
            }
            
    except Exception as e:
        # Gracefully handle any API errors
        return {
            "model_name": model_name,
            "response": None,
            "error": f"Error querying {model_name}: {str(e)}"
        }


def _query_openai(model: str, api_key: str, question: str, model_name: str) -> Dict[str, any]:
    """Helper function to query OpenAI models."""
    if not OPENAI_AVAILABLE:
        return {
            "model_name": model_name,
            "response": None,
            "error": "OpenAI package not installed"
        }
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Provide clear, concise answers."},
            {"role": "user", "content": question}
        ],
        max_tokens=1000,
        temperature=0.7
    )
    return {
        "model_name": model_name,
        "response": response.choices[0].message.content,
        "error": None
    }


def _query_anthropic(model: str, api_key: str, question: str, model_name: str) -> Dict[str, any]:
    """Helper function to query Anthropic Claude models."""
    if not ANTHROPIC_AVAILABLE:
        return {
            "model_name": model_name,
            "response": None,
            "error": "Anthropic package not installed"
        }
    client = Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=[
            {"role": "user", "content": question}
        ]
    )
    return {
        "model_name": model_name,
        "response": message.content[0].text,
        "error": None
    }


def _query_google(model: str, api_key: str, question: str, model_name: str) -> Dict[str, any]:
    """Helper function to query Google Gemini models."""
    if not GOOGLE_AVAILABLE:
        return {
            "model_name": model_name,
            "response": None,
            "error": "Google Generative AI package not installed"
        }
    genai.configure(api_key=api_key)
    model_instance = genai.GenerativeModel(model)
    response = model_instance.generate_content(question)
    return {
        "model_name": model_name,
        "response": response.text,
        "error": None
    }


def aggregate_answers(responses: List[Dict[str, any]], question: str) -> str:
    """
    Combines multiple AI responses into one clear, concise, coherent "super answer".
    
    This function:
    1. Filters out failed responses
    2. Identifies common themes and agreements
    3. Resolves contradictions by prioritizing consensus
    4. Creates a summarized, coherent final answer
    
    Args:
        responses: List of response dictionaries from AI models
        question: The original user question
        
    Returns:
        A single coherent answer combining all successful responses
    """
    # Step 1: Filter out failed responses and collect successful ones
    successful_responses = []
    failed_models = []
    
    for response_dict in responses:
        if response_dict.get("response") is not None:
            successful_responses.append(response_dict)
        else:
            failed_models.append(response_dict.get("model_name"))
            print(f"Warning: {response_dict.get('model_name')} failed: {response_dict.get('error')}")
    
    # If no successful responses, return error message
    if not successful_responses:
        return "Error: All AI models failed to respond. Please check your API keys and network connection."
    
    # If only one successful response, return it directly
    if len(successful_responses) == 1:
        return successful_responses[0]["response"]
    
    # Step 2: Use an AI model to synthesize the answers
    # We'll use OpenAI for synthesis (or the first available model)
    synthesis_model = None
    synthesis_api_key = None
    
    # Try to find an available model for synthesis
    for response_dict in successful_responses:
        model_name = response_dict["model_name"]
        if "openai" in model_name.lower():
            synthesis_model = "openai"
            synthesis_api_key = os.getenv("OPENAI_API_KEY")
            break
    
    # If no OpenAI available, use the first successful model's provider
    if not synthesis_model:
        first_model = successful_responses[0]["model_name"]
        for model_key, model_config in AI_MODELS.items():
            if model_key == first_model or first_model in model_key:
                synthesis_model = model_config["provider"]
                synthesis_api_key = os.getenv(model_config["api_key_env"])
                break
    
    # Step 3: Create synthesis prompt
    answers_text = "\n\n".join([
        f"Answer from {resp['model_name']}:\n{resp['response']}"
        for resp in successful_responses
    ])
    
    synthesis_prompt = f"""You are an expert at synthesizing multiple answers into one coherent response.

Original Question: {question}

Here are answers from {len(successful_responses)} different AI models:

{answers_text}

Please create a single, clear, concise, and coherent answer that:
1. Combines the best insights from all the answers above
2. Resolves any contradictions by identifying the most accurate information
3. Maintains a natural, flowing narrative
4. Is comprehensive yet concise
5. Prioritizes consensus where multiple models agree

Provide only the final synthesized answer, without mentioning that it came from multiple sources:"""

    # Step 4: Query synthesis model
    try:
        if synthesis_model == "openai" and synthesis_api_key and OPENAI_AVAILABLE:
            client = OpenAI(api_key=synthesis_api_key)
            synthesis_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at synthesizing information from multiple sources."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                max_tokens=1500,
                temperature=0.5
            )
            return synthesis_response.choices[0].message.content
        elif synthesis_model == "anthropic" and synthesis_api_key and ANTHROPIC_AVAILABLE:
            client = Anthropic(api_key=synthesis_api_key)
            message = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1500,
                messages=[
                    {"role": "user", "content": synthesis_prompt}
                ]
            )
            return message.content[0].text
        else:
            # Fallback: Simple concatenation if synthesis fails
            return _simple_aggregation(successful_responses)
    except Exception as e:
        print(f"Warning: Synthesis failed ({str(e)}). Using simple aggregation.")
        return _simple_aggregation(successful_responses)


def _simple_aggregation(responses: List[Dict[str, any]]) -> str:
    """
    Fallback aggregation method that combines answers when AI synthesis is unavailable.
    
    Args:
        responses: List of successful response dictionaries
        
    Returns:
        A combined answer string
    """
    combined = "Combined Answer from Multiple AI Models:\n\n"
    for i, resp in enumerate(responses, 1):
        combined += f"[{resp['model_name']}]:\n{resp['response']}\n\n"
    return combined.strip()


def query_all_models(question: str) -> List[Dict[str, any]]:
    """
    Queries all configured AI models in parallel for efficiency.
    
    Args:
        question: The user's question to send to all models
        
    Returns:
        List of response dictionaries from all models
    """
    responses = []
    
    # Use ThreadPoolExecutor to query all models in parallel
    with ThreadPoolExecutor(max_workers=len(AI_MODELS)) as executor:
        # Submit all queries
        future_to_model = {
            executor.submit(query_ai_model, model_name, model_config, question): model_name
            for model_name, model_config in AI_MODELS.items()
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                result = future.result()
                responses.append(result)
            except Exception as e:
                responses.append({
                    "model_name": model_name,
                    "response": None,
                    "error": f"Unexpected error: {str(e)}"
                })
    
    return responses


def main():
    """
    Main function that orchestrates the entire process:
    1. Gets user question
    2. Queries all AI models
    3. Aggregates responses
    4. Outputs final answer
    """
    print("=" * 60)
    print("AI Answer Aggregator")
    print("=" * 60)
    print("\nThis program queries multiple AI models and combines their answers.")
    print(f"Currently configured with {len(AI_MODELS)} models: {', '.join(AI_MODELS.keys())}\n")
    
    # Step 1: Get user question
    question = input("Enter your question: ").strip()
    
    if not question:
        print("Error: Question cannot be empty.")
        return
    
    print("\n" + "=" * 60)
    print("Querying AI models...")
    print("=" * 60)
    
    # Step 2: Query all AI models in parallel
    responses = query_all_models(question)
    
    # Display individual responses (optional, for transparency)
    print("\nIndividual Responses:")
    print("-" * 60)
    for resp in responses:
        if resp.get("response"):
            print(f"\n[{resp['model_name']}]:")
            print(resp["response"][:200] + "..." if len(resp["response"]) > 200 else resp["response"])
        else:
            print(f"\n[{resp['model_name']}]: Failed - {resp.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 60)
    print("Synthesizing answers...")
    print("=" * 60)
    
    # Step 3: Aggregate all responses into one coherent answer
    final_answer = aggregate_answers(responses, question)
    
    # Step 4: Output the final answer
    print("\n" + "=" * 60)
    print("FINAL SYNTHESIZED ANSWER")
    print("=" * 60)
    print(f"\n{final_answer}\n")
    print("=" * 60)


if __name__ == "__main__":
    main()

