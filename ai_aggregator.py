"""
AI Answer Aggregator - Enhanced Version
Fixed API calls, added error handling, timeouts, and improved aggregation.
"""

import os
import json
import signal
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from datetime import datetime
import time
import hashlib
from collections import defaultdict

# Third-party imports with graceful error handling
try:
    from anthropic import Anthropic  # type: ignore
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None  # type: ignore

try:
    import google.generativeai as genai  # type: ignore
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    genai = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None  # type: ignore

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type  # type: ignore
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    # Create dummy decorator and functions if tenacity not available
    def retry(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func
        return decorator
    
    def stop_after_attempt(n):  # type: ignore
        return None
    
    def wait_exponential(multiplier=1, min=2, max=10):  # type: ignore
        return None
    
    def retry_if_exception_type(*exceptions):  # type: ignore
        return None


# Model configuration
AI_MODELS = {
    "openai_gpt4o": {
        "provider": "openai",
        "model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
        "timeout": 30,
        "max_tokens": 2000
    },
    "openai_gpt41": {
        "provider": "openai", 
        "model": "gpt-4-turbo",
        "api_key_env": "OPENAI_API_KEY",
        "timeout": 30,
        "max_tokens": 2000
    },
    "anthropic_claude": {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet-20241022",
        "api_key_env": "ANTHROPIC_API_KEY",
        "timeout": 45,
        "max_tokens": 2000
    },
    "google_gemini": {
        "provider": "google",
        "model": "gemini-1.5-pro",
        "api_key_env": "GOOGLE_API_KEY",
        "timeout": 40,
        "max_tokens": 2000
    },
    "openai_gpt4o_mini": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "timeout": 20,
        "max_tokens": 2000
    }
}

# Cache for responses (simple in-memory cache)
response_cache = {}
CACHE_ENABLED = True
CACHE_TTL = 3600  # 1 hour


class TimeoutException(Exception):
    """Custom exception for timeout handling."""
    pass


def timeout_handler(signum, frame):
    """Handler for signal timeout."""
    raise TimeoutException("Query timed out")


def get_cache_key(model_name: str, question: str) -> str:
    """Generate cache key for question and model."""
    content = f"{model_name}:{question}"
    return hashlib.md5(content.encode()).hexdigest()


def validate_api_keys() -> List[str]:
    """Validate all required API keys are present."""
    missing_keys = []
    for model_name, config in AI_MODELS.items():
        key_env = config["api_key_env"]
        if not os.getenv(key_env):
            missing_keys.append(f"{model_name} ({key_env})")
    return missing_keys


# Apply retry decorator only if tenacity is available
if TENACITY_AVAILABLE:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    )
    def query_ai_model_with_retry(model_name: str, config: Dict, question: str) -> Dict[str, Any]:
        return _query_ai_model_impl(model_name, config, question)
else:
    def query_ai_model_with_retry(model_name: str, config: Dict, question: str) -> Dict[str, Any]:
        return _query_ai_model_impl(model_name, config, question)


def _query_ai_model_impl(model_name: str, config: Dict, question: str) -> Dict[str, Any]:
    """Query AI model implementation (used with or without retry)."""
    """
    Query a single AI model.
    
    Args:
        model_name: Name of the model (key in AI_MODELS)
        config: Model configuration dictionary
        question: User's question
        
    Returns:
        Dictionary containing response, metadata, and error information
    """
    provider = config["provider"]
    model = config["model"]
    key = os.getenv(config["api_key_env"])
    timeout = config.get("timeout", 30)
    
    # Check cache first
    if CACHE_ENABLED:
        cache_key = get_cache_key(model_name, question)
        if cache_key in response_cache:
            cached = response_cache[cache_key]
            if time.time() - cached["timestamp"] < CACHE_TTL:
                return cached["response"]
    
    if not key:
        return {
            "model_name": model_name,
            "response": None,
            "error": f"Missing API key: {config['api_key_env']}",
            "provider": provider,
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "cached": False
        }
    
    try:
        # Set timeout signal
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        start_time = time.time()
        
        if provider == "openai":
            result = _query_openai(model, key, question, model_name, config)
        elif provider == "anthropic":
            result = _query_anthropic(model, key, question, model_name, config)
        elif provider == "google":
            result = _query_google(model, key, question, model_name, config)
        else:
            result = {
                "model_name": model_name,
                "response": None,
                "error": f"Unknown provider: {provider}",
                "provider": provider,
                "model": model,
                "timestamp": datetime.now().isoformat(),
                "cached": False
            }
        
        elapsed_time = time.time() - start_time
        result["elapsed_time"] = elapsed_time
        
        # Cache successful response
        if result["response"] and CACHE_ENABLED:
            cache_key = get_cache_key(model_name, question)
            response_cache[cache_key] = {
                "response": result,
                "timestamp": time.time()
            }
        
        return result
        
    except TimeoutException as e:
        return {
            "model_name": model_name,
            "response": None,
            "error": f"Timeout after {timeout} seconds: {str(e)}",
            "provider": provider,
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": timeout,
            "cached": False
        }
    except Exception as e:
        return {
            "model_name": model_name,
            "response": None,
            "error": f"{type(e).__name__}: {str(e)}",
            "provider": provider,
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "cached": False
        }
    finally:
        # Disable alarm
        signal.alarm(0)


def _query_openai(model: str, api_key: str, question: str, model_name: str, config: Dict) -> Dict[str, Any]:
    """Query OpenAI models."""
    client = OpenAI(api_key=api_key)
    
    # FIXED: Using correct chat.completions endpoint
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        max_tokens=config.get("max_tokens", 2000),
        temperature=0.7,
        stream=False
    )
    
    return {
        "model_name": model_name,
        "response": response.choices[0].message.content,
        "error": None,
        "provider": "openai",
        "model": model,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        } if response.usage else None,
        "timestamp": datetime.now().isoformat(),
        "cached": False
    }


def _query_anthropic(model: str, api_key: str, question: str, model_name: str, config: Dict) -> Dict[str, Any]:
    """Query Anthropic models."""
    client = Anthropic(api_key=api_key)
    
    message = client.messages.create(
        model=model,
        max_tokens=config.get("max_tokens", 2000),
        temperature=0.7,
        system="You are a helpful assistant.",
        messages=[{"role": "user", "content": question}]
    )
    
    return {
        "model_name": model_name,
        "response": message.content[0].text,
        "error": None,
        "provider": "anthropic",
        "model": model,
        "usage": {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens
        } if hasattr(message, 'usage') else None,
        "timestamp": datetime.now().isoformat(),
        "cached": False
    }


def _query_google(model: str, api_key: str, question: str, model_name: str, config: Dict) -> Dict[str, Any]:
    """Query Google Gemini models."""
    genai.configure(api_key=api_key)
    model_instance = genai.GenerativeModel(
        model,
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": config.get("max_tokens", 2000),
        }
    )
    
    response = model_instance.generate_content(question)
    
    # Handle safety filters
    if response.prompt_feedback and response.prompt_feedback.block_reason:
        return {
            "model_name": model_name,
            "response": None,
            "error": f"Blocked by safety filters: {response.prompt_feedback.block_reason}",
            "provider": "google",
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "cached": False
        }
    
    return {
        "model_name": model_name,
        "response": response.text,
        "error": None,
        "provider": "google",
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "cached": False
    }


def analyze_responses(responses: List[Dict]) -> Dict[str, Any]:
    """Analyze responses for consensus and quality."""
    successful = [r for r in responses if r["response"]]
    failed = [r for r in responses if not r["response"]]
    
    # Calculate average response time
    avg_time = sum(r.get("elapsed_time", 0) for r in successful) / len(successful) if successful else 0
    
    # Token usage summary
    total_tokens = 0
    for r in successful:
        if r.get("usage"):
            if r["provider"] == "openai":
                total_tokens += r["usage"].get("total_tokens", 0)
            elif r["provider"] == "anthropic":
                total_tokens += r["usage"].get("input_tokens", 0) + r["usage"].get("output_tokens", 0)
    
    return {
        "total_models": len(responses),
        "successful_models": len(successful),
        "failed_models": len(failed),
        "success_rate": len(successful) / len(responses) if responses else 0,
        "average_response_time": avg_time,
        "total_tokens": total_tokens,
        "failed_details": failed
    }


def aggregate_answers(responses: List[Dict], question: str) -> Dict[str, Any]:
    """
    Aggregate answers from multiple models.
    
    Returns a dictionary with the final answer and metadata.
    """
    successful = [r for r in responses if r["response"]]
    failed = [r for r in responses if not r["response"]]
    
    if len(successful) == 0:
        return {
            "final_answer": "All models failed to provide a response.",
            "sources": [],
            "consensus_score": 0.0,
            "aggregation_method": "none",
            "metadata": {
                "total_models": len(responses),
                "successful_models": 0,
                "failed_models": len(failed)
            }
        }
    
    if len(successful) == 1:
        return {
            "final_answer": successful[0]["response"],
            "sources": [successful[0]["model_name"]],
            "consensus_score": 1.0,
            "aggregation_method": "single",
            "metadata": {
                "total_models": len(responses),
                "successful_models": 1,
                "failed_models": len(failed),
                "source_model": successful[0]["model_name"]
            }
        }
    
    # Multiple successful responses - use synthesis
    # Prepare combined input for synthesis
    combined = "\n\n".join(
        f"## Response from {r['model_name']} ({r['provider']}):\n{r['response']}"
        for r in successful
    )
    
    # Try to get consensus first (simple text similarity)
    consensus_score = calculate_consensus([r["response"] for r in successful])
    
    # Use GPT-4o for synthesis
    try:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            # Fallback to concatenation if no OpenAI key
            return {
                "final_answer": f"Multiple responses received:\n\n{combined}",
                "sources": [r["model_name"] for r in successful],
                "consensus_score": consensus_score,
                "aggregation_method": "concatenation",
                "metadata": {
                    "total_models": len(responses),
                    "successful_models": len(successful),
                    "failed_models": len(failed),
                    "consensus_score": consensus_score
                }
            }
        
        client = OpenAI(api_key=key)
        
        synthesis_prompt = f"""
You are an expert synthesizer. Combine the following responses into one comprehensive, clear answer.

Original question: {question}

Responses from different AI models:
{combined}

Instructions:
1. Identify the key points from all responses
2. Resolve any contradictions (note if consensus exists or highlight differences)
3. Provide a unified, well-structured answer
4. Include a brief "Sources" section mentioning which models contributed
5. Keep the tone professional and neutral
6. If there are significant disagreements, mention them

Synthesized answer:
"""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0.3,
            max_tokens=3000
        )
        
        final_answer = response.choices[0].message.content
        
        return {
            "final_answer": final_answer,
            "sources": [r["model_name"] for r in successful],
            "consensus_score": consensus_score,
            "aggregation_method": "synthesis",
            "metadata": {
                "total_models": len(responses),
                "successful_models": len(successful),
                "failed_models": len(failed),
                "consensus_score": consensus_score,
                "synthesis_model": "gpt-4o",
                "synthesis_tokens": response.usage.total_tokens if response.usage else None
            }
        }
        
    except Exception as e:
        # Fallback to best response selection
        best_response = select_best_response(successful)
        return {
            "final_answer": best_response["response"],
            "sources": [best_response["model_name"]],
            "consensus_score": consensus_score,
            "aggregation_method": "best_of",
            "metadata": {
                "total_models": len(responses),
                "successful_models": len(successful),
                "failed_models": len(failed),
                "consensus_score": consensus_score,
                "selected_model": best_response["model_name"],
                "fallback_reason": str(e)
            }
        }


def calculate_consensus(responses: List[str]) -> float:
    """Calculate a simple consensus score based on common keywords."""
    if len(responses) <= 1:
        return 1.0
    
    # Simple consensus calculation - count overlapping significant words
    all_words = []
    for resp in responses:
        words = set(resp.lower().split()[:50])  # First 50 words
        all_words.append(words)
    
    # Calculate Jaccard similarity between all pairs
    similarities = []
    for i in range(len(all_words)):
        for j in range(i + 1, len(all_words)):
            intersection = len(all_words[i] & all_words[j])
            union = len(all_words[i] | all_words[j])
            if union > 0:
                similarity = intersection / union
                similarities.append(similarity)
    
    return sum(similarities) / len(similarities) if similarities else 0.0


def select_best_response(responses: List[Dict]) -> Dict:
    """Select the best response based on heuristic rules."""
    # Prefer models that typically give better answers
    model_priority = {
        "openai_gpt4o": 10,
        "anthropic_claude": 9,
        "openai_gpt41": 8,
        "google_gemini": 7,
        "openai_gpt4o_mini": 6
    }
    
    # Score each response
    scored = []
    for resp in responses:
        score = model_priority.get(resp["model_name"], 5)
        
        # Add points for response length (not too short, not too long)
        response_length = len(resp["response"])
        if 100 <= response_length <= 2000:
            score += 2
        elif response_length > 2000:
            score += 1
        
        # Add points for faster responses
        if resp.get("elapsed_time", 0) < 10:
            score += 1
        
        scored.append((score, resp))
    
    # Return highest scored response
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def query_all_models(question: str, use_cache: bool = True) -> List[Dict]:
    """
    Query all configured AI models in parallel.
    
    Args:
        question: The question to ask
        use_cache: Whether to use cached responses
        
    Returns:
        List of response dictionaries from all models
    """
    global CACHE_ENABLED
    original_cache_setting = CACHE_ENABLED
    CACHE_ENABLED = use_cache
    
    responses = []
    
    with ThreadPoolExecutor(max_workers=min(5, len(AI_MODELS))) as executor:
        # Submit all tasks
        future_to_model = {
            executor.submit(query_ai_model_with_retry, name, cfg, question): name
            for name, cfg in AI_MODELS.items()
        }
        
        # Collect results as they complete
        completed = 0
        total = len(future_to_model)
        
        for future in as_completed(future_to_model):
            model_name = future_to_model[future]
            completed += 1
            
            try:
                result = future.result(timeout=AI_MODELS[model_name].get("timeout", 30) + 5)
                responses.append(result)
                print(f"[{completed}/{total}] {model_name}: {'✓' if result['response'] else '✗'}")
                
            except FutureTimeoutError:
                responses.append({
                    "model_name": model_name,
                    "response": None,
                    "error": "Future timeout exceeded",
                    "provider": AI_MODELS[model_name]["provider"],
                    "model": AI_MODELS[model_name]["model"],
                    "timestamp": datetime.now().isoformat(),
                    "cached": False
                })
                print(f"[{completed}/{total}] {model_name}: ✗ (timeout)")
            except Exception as e:
                responses.append({
                    "model_name": model_name,
                    "response": None,
                    "error": f"Execution error: {str(e)}",
                    "provider": AI_MODELS[model_name]["provider"],
                    "model": AI_MODELS[model_name]["model"],
                    "timestamp": datetime.now().isoformat(),
                    "cached": False
                })
                print(f"[{completed}/{total}] {model_name}: ✗ (error)")
    
    CACHE_ENABLED = original_cache_setting
    return responses


def save_results_to_file(question: str, responses: List[Dict], final_answer: Dict, filename: str = None):
    """Save query results to a JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ai_aggregator_results_{timestamp}.json"
    
    results = {
        "question": question,
        "timestamp": datetime.now().isoformat(),
        "responses": responses,
        "analysis": analyze_responses(responses),
        "final_answer": final_answer
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return filename


def display_results(question: str, responses: List[Dict], final_answer: Dict):
    """Display results in a formatted way."""
    print("\n" + "="*80)
    print("AI ANSWER AGGREGATOR - RESULTS")
    print("="*80)
    
    print(f"\nQuestion: {question}")
    
    # Analysis
    analysis = analyze_responses(responses)
    print(f"\n📊 ANALYSIS:")
    print(f"  • Models queried: {analysis['total_models']}")
    print(f"  • Successful: {analysis['successful_models']}")
    print(f"  • Failed: {analysis['failed_models']}")
    print(f"  • Success rate: {analysis['success_rate']:.1%}")
    print(f"  • Avg response time: {analysis['average_response_time']:.2f}s")
    if analysis['total_tokens'] > 0:
        print(f"  • Estimated tokens used: {analysis['total_tokens']}")
    
    # Individual responses
    print(f"\n📝 INDIVIDUAL RESPONSES:")
    for r in responses:
        status = "✓" if r["response"] else "✗"
        time_str = f" ({r.get('elapsed_time', 0):.1f}s)" if r.get('elapsed_time') else ""
        cached_str = " [cached]" if r.get('cached', False) else ""
        
        print(f"  {status} {r['model_name']}{time_str}{cached_str}")
        if not r["response"] and r.get("error"):
            print(f"    Error: {r['error']}")
    
    # Final answer
    print(f"\n🎯 FINAL ANSWER ({final_answer['aggregation_method'].upper()}):")
    print("-" * 40)
    print(final_answer["final_answer"])
    print("-" * 40)
    
    # Metadata
    print(f"\n📋 METADATA:")
    print(f"  • Sources: {', '.join(final_answer['sources'])}")
    print(f"  • Consensus score: {final_answer['consensus_score']:.2f}")
    print(f"  • Aggregation method: {final_answer['aggregation_method']}")
    
    if final_answer['metadata'].get('synthesis_tokens'):
        print(f"  • Synthesis tokens: {final_answer['metadata']['synthesis_tokens']}")
    
    print("\n" + "="*80)


def main():
    """Main function."""
    global AI_MODELS
    
    print("🤖 AI ANSWER AGGREGATOR - ENHANCED VERSION")
    print("="*50)
    print(f"Available models: {', '.join(AI_MODELS.keys())}")
    print("="*50)
    
    # Validate API keys
    missing_keys = validate_api_keys()
    if missing_keys:
        print("\n⚠️  WARNING: Missing API keys for:")
        for key in missing_keys:
            print(f"   • {key}")
        print("\nModels with missing keys will fail.")
        proceed = input("\nContinue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            return
    
    # Get question
    print("\n" + "-"*50)
    question = input("\nEnter your question: ").strip()
    if not question:
        print("Empty question. Exiting.")
        return
    
    # Query options
    print("\nOptions:")
    print("1. Query all models (default)")
    print("2. Query specific models")
    choice = input("Choose option (1-2): ").strip() or "1"
    
    models_to_query = AI_MODELS
    if choice == "2":
        print("\nAvailable models:")
        for i, model in enumerate(AI_MODELS.keys(), 1):
            print(f"{i}. {model}")
        selections = input("\nEnter model numbers (comma-separated, e.g., 1,3,5): ").strip()
        try:
            selected_indices = [int(x.strip()) - 1 for x in selections.split(",") if x.strip()]
            selected_models = list(AI_MODELS.keys())
            models_to_query = {selected_models[i]: AI_MODELS[selected_models[i]] 
                              for i in selected_indices if 0 <= i < len(selected_models)}
            if not models_to_query:
                print("No valid models selected. Using all models.")
                models_to_query = AI_MODELS
        except:
            print("Invalid selection. Using all models.")
    
    # Cache option
    use_cache = input("\nUse cached responses if available? (y/n, default=y): ").strip().lower()
    use_cache = use_cache != 'n'
    
    # Query models
    # Temporarily replace AI_MODELS with selected models
    original_models = AI_MODELS
    AI_MODELS = models_to_query
    
    print(f"\n{' Querying models... ':=^50}")
    print(f"Question: {question[:80]}{'...' if len(question) > 80 else ''}")
    print(f"Models: {', '.join(models_to_query.keys())}")
    print(f"Caching: {'Enabled' if use_cache else 'Disabled'}")
    print("="*50 + "\n")
    
    try:
        start_time = time.time()
        responses = query_all_models(question, use_cache=use_cache)
        query_time = time.time() - start_time
        
        # Aggregate answers
        print(f"\n{' Synthesizing responses... ':=^50}")
        final_answer = aggregate_answers(responses, question)
        
        # Display results
        display_results(question, responses, final_answer)
        
        # Save option
        save_option = input("\n💾 Save results to file? (y/n): ").strip().lower()
        if save_option == 'y':
            filename = save_results_to_file(question, responses, final_answer)
            print(f"Results saved to: {filename}")
        
        print(f"\n⏱️  Total execution time: {query_time:.2f} seconds")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
    finally:
        # Restore original models
        AI_MODELS = original_models


if __name__ == "__main__":
    main()