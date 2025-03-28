import os
import streamlit as st
import random
from typing import Tuple, Dict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from atla import Atla, AsyncAtla
from dotenv import load_dotenv
import asyncio

load_dotenv(dotenv_path="/.env")

# Set page config
st.set_page_config(page_title="Meta-GPT", layout="wide")

# Configuration parameters
QUALITY_THRESHOLD = 4.0  # Threshold for acceptable response quality
MAX_ITERATIONS = 3  # Maximum number of refinement iterations

# Split the evaluation prompt into separate dimensions
ACCURACY_PROMPT = """
    Evaluate the response on Accuracy: Is the response factually correct, free from even minor inaccuracies, and demonstrating a deep, nuanced understanding of the subject matter?

    Scoring Rubric:
    Score 1: The response contains any factual errors, no matter how minor, or shows any signs of hallucination.
    Score 2: The response is mostly accurate but lacks precision in technical details or contains slight oversimplifications.
    Score 3: The response is accurate and precise, but fails to address potential edge cases or exceptions.
    Score 4: The response is highly accurate, addresses edge cases, but doesn't demonstrate extraordinary depth of knowledge.
    Score 5: The response is impeccably accurate, demonstrates expert-level understanding, and provides insights beyond common knowledge.

    Provide:
    - A numeric score (1-5, where 5 is near impossible to achieve)
    - A detailed critique justifying the score, highlighting even minor inaccuracies
    - Specific suggestions for improvement, including additional facts or nuances that could have been included
"""

RELEVANCE_PROMPT = """
    Evaluate the response on Relevance: Does the response answer the user's question with laser-focused precision, anticipating and addressing all possible interpretations and implications?

    Scoring Rubric:
    Score 1: The response fails to directly address the core question or misses any subtext or implicit aspects.
    Score 2: The response addresses the main question but overlooks subtle nuances or related concerns.
    Score 3: The response is relevant and comprehensive but fails to prioritize the most critical aspects of the question.
    Score 4: The response is highly relevant, prioritizes well, but doesn't explore all possible interpretations of the question.
    Score 5: The response demonstrates perfect relevance, addresses all explicit and implicit aspects, and provides valuable additional context.

    Provide:
    - A numeric score (1-5, where 5 is near impossible to achieve)
    - A detailed critique justifying the score, analyzing how well each part of the question was addressed
    - Specific suggestions for improvement, including unexplored angles or interpretations of the question
"""

CLARITY_PROMPT = """
    Evaluate the response on Clarity: Is the response structured with perfect logical flow, using precise language that leaves no room for misinterpretation?

    Scoring Rubric:
    Score 1: The response has any structural issues, unclear transitions, or imprecise language use.
    Score 2: The response is generally clear but contains minor ambiguities or could be more concise.
    Score 3: The response is well-structured and clear, but lacks optimal organization for the subject matter.
    Score 4: The response demonstrates excellent clarity and structure, but falls short of absolute perfection in precision.
    Score 5: The response exhibits flawless organization, crystal-clear explanations, and language so precise it could serve as a technical reference.

    Provide:
    - A numeric score (1-5, where 5 is near impossible to achieve)
    - A detailed critique justifying the score, analyzing sentence structure, word choice, and overall organization
    - Specific suggestions for improvement, including restructuring ideas or refining language for ultimate clarity
"""

DEPTH_PROMPT = """
    Evaluate the response on Depth: Does the response provide extraordinarily comprehensive coverage, offering cutting-edge insights and exploring the topic to its fullest extent?

    Scoring Rubric:
    Score 1: The response lacks depth, misses key concepts, or fails to go beyond surface-level information.
    Score 2: The response provides good coverage but doesn't delve into advanced concepts or implications.
    Score 3: The response offers solid depth with some advanced concepts, but doesn't push the boundaries of the topic.
    Score 4: The response provides excellent depth, touching on cutting-edge ideas, but falls short of exhaustive coverage.
    Score 5: The response demonstrates unparalleled depth, offering groundbreaking insights, and exhaustively covering all aspects including future implications.

    Provide:
    - A numeric score (1-5, where 5 is near impossible to achieve)
    - A detailed critique justifying the score, analyzing the breadth and depth of concepts covered
    - Specific suggestions for improvement, including additional advanced topics, interdisciplinary connections, or futuristic implications that could have been explored
"""

# Initialize API keys from environment variables or Streamlit secrets
def initialize_api_keys():
    # Load from .env file (already done via load_dotenv() at the top of your script)
    # No need to check for Streamlit secrets if you're using .env exclusively
    
    # Check if required keys are in environment variables
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TOGETHER_API_KEY", "ATLA_API_KEY"]
    missing_keys = [key for key in required_keys if not os.environ.get(key)]
    
    if missing_keys:
        st.sidebar.error(f"Missing API keys: {', '.join(missing_keys)}")
        st.sidebar.info("Please add these keys to your .env file")
        return False
    
    return True


# Initialize models and session state
def initialize_app():
    keys_loaded = initialize_api_keys()

    # Initialize session state variables if they don't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            SystemMessage(
                content="You are a helpful assistant that can answer questions and help with tasks."
            )
        ]
        
    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None
        
    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    # Only initialize models if keys are loaded and not already initialized
    if not st.session_state.initialized and keys_loaded:
        try:
            st.session_state.gpt4o = init_chat_model("gpt-4o", model_provider="openai")
            st.session_state.claude = init_chat_model(
                "claude-3-7-sonnet-20250219", model_provider="anthropic"
            )
            st.session_state.deepseek = init_chat_model(
                "deepseek-ai/DeepSeek-V3", model_provider="together"
            )
            st.session_state.atla = Atla()
            st.session_state.async_atla = AsyncAtla()
            st.session_state.initialized = True

        except Exception as e:
            st.error(f"Error initializing models: {e}")
            st.warning("Please check your API keys in the .env file.")
            st.session_state.initialized = False


async def evaluate_dimension(question: str, response: str, dimension_prompt: str) -> Tuple[float, str]:
    """Evaluate a single dimension using Atla's Selene model asynchronously."""
    eval_response = await st.session_state.async_atla.evaluation.create(
        model_id="atla-selene",
        model_input=question,
        model_output=response,
        evaluation_criteria=dimension_prompt,
    )
    evaluation = eval_response.result.evaluation
    return float(evaluation.score), evaluation.critique


async def evaluate_with_atla_async(inputs: dict[str, str]) -> Tuple[float, Dict[str, Dict]]:
    """Evaluate response using Atla's Selene model across all dimensions asynchronously."""
    # Create tasks for all dimensions
    accuracy_task = evaluate_dimension(inputs["question"], inputs["response"], ACCURACY_PROMPT)
    relevance_task = evaluate_dimension(inputs["question"], inputs["response"], RELEVANCE_PROMPT)
    clarity_task = evaluate_dimension(inputs["question"], inputs["response"], CLARITY_PROMPT)
    depth_task = evaluate_dimension(inputs["question"], inputs["response"], DEPTH_PROMPT)
    
    # Run all evaluations concurrently using asyncio.gather
    accuracy_result, relevance_result, clarity_result, depth_result = await asyncio.gather(
        accuracy_task, relevance_task, clarity_task, depth_task
    )
    
    # Unpack results
    accuracy_score, accuracy_critique = accuracy_result
    relevance_score, relevance_critique = relevance_result
    clarity_score, clarity_critique = clarity_result
    depth_score, depth_critique = depth_result
    
    # Calculate average score
    avg_score = (accuracy_score + relevance_score + clarity_score + depth_score) / 4
    
    # Compile detailed results
    detailed_results = {
        "accuracy": {"score": accuracy_score, "critique": accuracy_critique},
        "relevance": {"score": relevance_score, "critique": relevance_critique},
        "clarity": {"score": clarity_score, "critique": clarity_critique},
        "depth": {"score": depth_score, "critique": depth_critique}
    }
    
    # Compile overall critique
    overall_critique = f"""
    Accuracy ({accuracy_score}/5): {accuracy_critique}
    
    Relevance ({relevance_score}/5): {relevance_critique}
    
    Clarity ({clarity_score}/5): {clarity_critique}
    
    Depth ({depth_score}/5): {depth_critique}
    
    **Overall Score: {avg_score:.2f}/5**
    """
    
    return avg_score, overall_critique, detailed_results


def evaluate_response(question: str, response: str) -> Dict:
    """Evaluate a single response using Selene."""
    inputs = {"question": question, "response": response}
    # Use asyncio to run the async function
    score, critique, detailed_results = asyncio.run(evaluate_with_atla_async(inputs))
    return {"score": score, "critique": critique, "detailed_results": detailed_results}


def get_responses(
    question: str, feedback: str = "", with_status: bool = True
) -> Dict[str, str]:
    """Get responses from all LLMs for a given question."""
    st.session_state.chat_messages.append(HumanMessage(content=question))
    if feedback:
        st.session_state.chat_messages.append(HumanMessage(content=feedback))
    responses = {}

    if with_status:
        # Create progress trackers for each model
        with st.status(
            "Generating responses from all models...", expanded=True
        ) as status:
            # Get response from GPT-4o
            status.update(label="Getting response from GPT-4o...")
            gpt_response = st.session_state.gpt4o.invoke(st.session_state.chat_messages)
            responses["GPT-4o"] = gpt_response.content

            # Get response from Claude
            status.update(label="Getting response from Claude 3.7...")
            claude_response = st.session_state.claude.invoke(
                st.session_state.chat_messages
            )
            responses["Claude 3.7"] = claude_response.content

            # Get response from DeepSeek
            status.update(label="Getting response from DeepSeekV3.0...")
            deepseek_response = st.session_state.deepseek.invoke(
                st.session_state.chat_messages
            )
            responses["DeepSeekV3.0"] = deepseek_response.content

            status.update(label="All responses generated successfully!", state="complete")
    else:
        # Get responses without status bar (for refinement)
        st.write("Getting response from models...")

        # Get response from GPT-4o
        gpt_response = st.session_state.gpt4o.invoke(st.session_state.chat_messages)
        responses["GPT-4o"] = gpt_response.content

        # Get response from Claude
        claude_response = st.session_state.claude.invoke(st.session_state.chat_messages)
        responses["Claude 3.7"] = claude_response.content

        # Get response from DeepSeek
        deepseek_response = st.session_state.deepseek.invoke(
            st.session_state.chat_messages
        )
        responses["DeepSeekV3.0"] = deepseek_response.content

    return responses


def evaluate_all_responses(
    question: str, responses: Dict[str, str], use_status: bool = True
) -> Dict[str, Dict]:
    """Evaluate all responses and return their evaluations."""
    evaluations = {}

    if (
        use_status and len(st.session_state.chat_history) <= 1
    ):  # Only use status on initial response
        with st.status("Evaluating responses with Selene...", expanded=True) as status:
            for model_name, response in responses.items():
                status.update(label=f"Evaluating {model_name} response...")
                evaluation = evaluate_response(question, response)
                evaluations[model_name] = evaluation

            status.update(label="All evaluations complete!", state="complete")
    else:
        # Simple version without status
        st.write("Evaluating responses with Selene...")
        for model_name, response in responses.items():
            evaluation = evaluate_response(question, response)
            evaluations[model_name] = evaluation
        st.write("All evaluations complete!")

    return evaluations


def select_best_response(evaluations: Dict[str, Dict]) -> Tuple[str, Dict]:
    """Select the best response based on overall score. Randomly choose if tied."""
    best_score = -1
    tied_models = []

    for model_name, evaluation in evaluations.items():
        overall_score = evaluation["score"]

        if overall_score > best_score:
            # New highest score - clear previous ties and start fresh
            best_score = overall_score
            tied_models = [(model_name, evaluation)]
        elif overall_score == best_score:
            # Tie detected - add to the list of tied models
            tied_models.append((model_name, evaluation))

    # If there are multiple models tied for the highest score, randomly select one
    if tied_models:
        best_model, best_evaluation = random.choice(tied_models)

    return best_model, best_evaluation


def refine_responses(question: str, model: str, evaluation: Dict) -> Tuple[str, Dict]:
    """Refine a response based on Selene's critique."""
    critique = evaluation["critique"]
    feedback = f"Please improve your previous response based on this feedback: {critique}"

    # Display refining message
    st.write(f"Refining response with {model}...")

    # Get improved responses without status bar (to avoid nesting)
    improved_responses = get_responses(question, feedback, with_status=False)
    improved_response = improved_responses[model]

    # Re-evaluate the improved response
    st.write("Re-evaluating refined response...")
    new_evaluation = evaluate_response(question, improved_response)

    st.write("Refinement complete!")

    return improved_response, new_evaluation


def meta_chat(question: str) -> Dict:
    """Process user question through the Meta-GPT system."""
    iteration = 0
    refinement_history = []

    # Step 1: Get initial responses from all models
    responses = get_responses(question)

    # Step 2: Evaluate all responses
    # Use status only for the first message
    evaluations = evaluate_all_responses(
        question, responses, use_status=len(st.session_state.chat_history) <= 1
    )

    # Step 3: Select best response
    best_model, best_evaluation = select_best_response(evaluations)
    best_response = responses[best_model]
    st.session_state.chat_messages.append(AIMessage(content=best_response))
    best_score = best_evaluation["score"]

    # Record initial state
    refinement_history.append(
        {
            "iteration": iteration,
            "model": best_model,
            "response": best_response,
            "evaluation": best_evaluation,
            "score": best_score,
        }
    )

    # Step 4: Iterative refinement if score is below threshold
    while best_score < QUALITY_THRESHOLD and iteration < MAX_ITERATIONS:
        iteration += 1
        st.info(
            f"Response quality ({best_score:.2f}/5) below threshold ({QUALITY_THRESHOLD}/5). Refining..."
        )

        # Refine the best response based on feedback
        improved_response, new_evaluation = refine_responses(
            question, best_model, best_evaluation
        )
        new_score = new_evaluation["score"]

        # Update best response if improved
        if new_score > best_score:
            best_response = improved_response
            best_evaluation = new_evaluation
            best_score = new_score
            # Update the AI message in chat_messages
            st.session_state.chat_messages[-1] = AIMessage(content=best_response)

        # Record refinement state
        refinement_history.append(
            {
                "iteration": iteration,
                "model": best_model,
                "response": improved_response,
                "evaluation": new_evaluation,
                "score": new_score,
            }
        )

    # Step 5: Return final result
    result = {
        "question": question,
        "best_model": best_model,
        "best_response": best_response,
        "best_score": best_score,
        "iterations_required": iteration,
        "all_evaluations": evaluations,
        "refinement_history": refinement_history,
        "threshold_met": best_score >= QUALITY_THRESHOLD,
        "all_initial_responses": responses,
    }

    return result


def display_chat():
    """Display the chat interface and history."""
    # Display chat history
    for entry in st.session_state.chat_history:
        if entry["role"] == "user":
            with st.chat_message("user"):
                st.markdown(entry["content"])
        else:
            # Use just "assistant" for avatar to avoid errors
            with st.chat_message("assistant"):
                st.markdown(entry["content"])

                # Add a footnote with model and score info
                st.caption(f"{entry['model']} (Score: {entry['score']:.2f}/5)")


def display_evaluation_details():
    """Display detailed evaluation information."""
    if st.session_state.latest_result:
        result = st.session_state.latest_result

        # Display best model and score
        st.subheader(f"Best Model: {result['best_model']}")
        st.metric("Overall Score", f"{result['best_score']:.2f}/5")

        # Refinement information
        if result["iterations_required"] > 0:
            st.subheader("Refinement Process")
            st.write(
                f"Required {result['iterations_required']} refinements to reach quality threshold."
            )

            # Create tabs for each refinement iteration
            tabs = st.tabs(
                ["Initial"]
                + [f"Refinement {i+1}" for i in range(result["iterations_required"])]
            )

            for i, tab in enumerate(tabs):
                if i < len(result["refinement_history"]):
                    refinement = result["refinement_history"][i]
                    with tab:
                        st.metric("Score", f"{refinement['score']:.2f}/5")

                        st.write("**Response:**")
                        st.text_area(
                            "Response Text",
                            value=refinement["response"],
                            height=150,
                            key=f"refinement_response_{i}",
                            disabled=True,
                        )

                        st.write("**Atla Critique's across different dimensions:**")
                        st.write(refinement["evaluation"]["critique"])

        # Model comparison
        st.subheader("Model Comparison")
        for model, eval_data in result["all_evaluations"].items():
            with st.expander(f"{model}: {eval_data['score']:.2f}/5"):
                st.write("**Initial Response:**")
                st.text_area(
                    "Response",
                    value=result["all_initial_responses"][model],
                    height=150,
                    key=f"response_{model}",
                    disabled=True,
                )

                st.write("**Atla Critique's across different dimensions:**")
                st.write(eval_data["critique"])


def main():
    """Main app function"""
    # Initialize the app
    initialize_app()

    # Initialize session state for sidebar visibility if not exists
    if "show_analysis" not in st.session_state:
        st.session_state.show_analysis = False

    # Main content takes full width when analysis is collapsed
    if st.session_state.get("latest_result") and st.session_state.show_analysis:
        col1, col2 = st.columns([2, 1])
    else:
        # Use full width for main content when analysis is collapsed
        col1 = st.container()
        col2 = None  # We won't use col2 when analysis is collapsed

    with col1:
        # Display header
        st.title("🤖 Meta-GPT")
        st.markdown(
            """
        This app uses multiple LLMs (GPT-4o, Claude 3.7, and DeepSeekV3.0) to answer your questions.
        The world's best LLM-as-a-Judge, [Selene](https://www.atla-ai.com/api), evaluates each response, and the best one is selected and refined if needed.
        """
        )

        # Add toggle for analysis panel if we have results
        if st.session_state.get("latest_result"):
            toggle_col1, toggle_col2 = st.columns([4, 1])
            with toggle_col2:
                if st.button(
                    "📊 "
                    + (
                        "Hide Analysis"
                        if st.session_state.show_analysis
                        else "Show Analysis"
                    )
                ):
                    st.session_state.show_analysis = not st.session_state.show_analysis
                    st.rerun()

        # Display chat interface
        display_chat()

        # Check if API keys are configured
        if not st.session_state.get("initialized", False):
            st.warning("Please configure your API keys in the sidebar to continue.")
            return

        # Chat input
        user_input = st.chat_input("Ask a question...")

    # Use a separate column for evaluation details
    if (
        st.session_state.get("latest_result")
        and st.session_state.show_analysis
        and col2 is not None
    ):
        with col2:
            st.title("Response Analysis")
            display_evaluation_details()

    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Add to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Get meta chat response
        with st.spinner("Processing your question..."):
            result = meta_chat(user_input)

        # Store latest result for sidebar display
        st.session_state.latest_result = result

        # Auto-expand the analysis panel when a new response comes in
        st.session_state.show_analysis = True

        # Display assistant message
        with st.chat_message("assistant"):
            st.markdown(result["best_response"])
            st.caption(f"{result['best_model']} (Score: {result['best_score']:.2f}/5)")

        # Add to history
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result["best_response"],
                "model": result["best_model"],
                "score": result["best_score"],
            }
        )

        # Force a refresh to update the evaluation details
        st.rerun()


if __name__ == "__main__":
    main()
