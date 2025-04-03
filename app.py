import streamlit as st
import os
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
from PIL import Image
import io
import requests
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
stability_api_key = os.getenv("STABILITY_API_KEY")

# List of supported models with detailed information
MODELS = {
    "llama-3.1-8b-instant": {
        "description": "Fast, efficient model for quick responses",
        "category": "General",
        "token_limit": 16384,
        "strengths": "Speed, efficiency",
        "best_for": "Quick conversations, basic tasks"
    },
    "deepseek-r1-distill-llama-70b": {
        "description": "Advanced distilled model with excellent performance",
        "category": "Advanced",
        "token_limit": 32768,
        "strengths": "Knowledge, reasoning",
        "best_for": "Complex reasoning, detailed explanations"
    },
    "qwen-2.5-32b": {
        "description": "High-quality model for detailed responses",
        "category": "Advanced",
        "token_limit": 32768,
        "strengths": "Quality, context handling",
        "best_for": "Longer conversations, nuanced responses"
    },
    "llama-3.3-70b-specdec": {
        "description": "Speculative decoding-enhanced Llama 3 for faster responses",
        "category": "Specialized",
        "token_limit": 16384,
        "strengths": "Speed, English tasks, technical content",
        "best_for": "Fast English generation, coding assistance"
    },
    "qwen-2.5-coder-32b": {
        "description": "Model optimized for coding tasks",
        "category": "Specialized",
        "token_limit": 32768,
        "strengths": "Code generation, technical knowledge",
        "best_for": "Programming assistance, technical documentation"
    },
    "stable-diffusion-xl": {
        "description": "Stability AI's advanced image generation model",
        "category": "Image Generation",
        "token_limit": 0,  # Not applicable for image models
        "strengths": "High-quality image generation, creative visuals",
        "best_for": "Generating images from text prompts"
    }
}

# Model categories for organization
MODEL_CATEGORIES = {
    "General": ["llama-3.1-8b-instant"],
    "Advanced": ["deepseek-r1-distill-llama-70b", "qwen-2.5-32b"],
    "Specialized": ["llama-3.3-70b-specdec", "qwen-2.5-coder-32b"],
    "Image Generation": ["stable-diffusion-xl"]
}

# System prompts for different conversation modes
SYSTEM_PROMPTS = {
    "General": "You are a helpful, harmless, and honest AI assistant.",
    "Creative": "You are a creative AI assistant that helps with brainstorming, storytelling, and creative writing. Be imaginative and inspirational in your responses.",
    "Technical": "You are a technical AI assistant that specializes in programming, data analysis, and technical topics. Provide detailed and accurate technical information.",
    "Concise": "You are a concise AI assistant. Provide brief, to-the-point responses that get straight to the answer without unnecessary elaboration.",
    "Multimodal": "You are a multimodal AI assistant that can handle both text and image generation requests. When asked to create images, provide detailed prompts for the image generation model."
}

def initialize_session_state():
    """Initialize all session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'model' not in st.session_state:
        st.session_state.model = list(MODELS.keys())[0]
    if 'memory_length' not in st.session_state:
        st.session_state.memory_length = 5
    if 'system_prompt' not in st.session_state:
        st.session_state.system_prompt = SYSTEM_PROMPTS["General"]
    if 'conversation_mode' not in st.session_state:
        st.session_state.conversation_mode = "General"
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7
    if 'max_tokens' not in st.session_state:
        st.session_state.max_tokens = 1000
    if 'session_start_time' not in st.session_state:
        st.session_state.session_start_time = datetime.now()
    if 'usage_stats' not in st.session_state:
        st.session_state.usage_stats = {
            "messages_sent": 0,
            "tokens_used": 0,
            "models_used": {},
            "response_speeds": [],
            "images_generated": 0
        }
    if 'saved_chats' not in st.session_state:
        st.session_state.saved_chats = []
    if 'current_chat_title' not in st.session_state:
        st.session_state.current_chat_title = f"Chat {len(st.session_state.saved_chats) + 1}"
    if 'image_generation_params' not in st.session_state:
        st.session_state.image_generation_params = {
            "width": 1024,
            "height": 1024,
            "steps": 30,
            "cfg_scale": 7.0,
            "sampler": "K_DPMPP_2M",
            "style_preset": None,
            "seed": None
        }
    if 'generated_images' not in st.session_state:
        st.session_state.generated_images = []

def create_conversation(model, memory_length, system_prompt, temperature, max_tokens):
    """Create a new conversation with the specified parameters"""
    memory = ConversationBufferWindowMemory(k=memory_length)
    
    # Preload existing chat history into memory
    for message in st.session_state.chat_history:
        memory.save_context({'input': message['human']}, {'output': message['AI']})
    
    # Skip LLM initialization for image generation model
    if model == "stable-diffusion-xl":
        return None
    
    # Initialize Groq chat model with parameters
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Create a proper prompt template with the system prompt
    template = f"{system_prompt}\n\nCurrent conversation:\n{{history}}\nHuman: {{input}}\nAI: "
    prompt_template = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )
    
    # Create conversation chain with system prompt
    return ConversationChain(
        llm=groq_chat,
        memory=memory,
        prompt=prompt_template
    )

def calculate_response_speed(response_text, response_time):
    """Calculate the response speed in tokens/second"""
    # Estimate tokens (improved estimate - approximately 4 chars per token)
    estimated_tokens = len(response_text) / 4
    
    # Calculate speed (tokens per second)
    if response_time > 0:
        speed = estimated_tokens / response_time
        return speed
    return 0

def generate_image(prompt, negative_prompt=None):
    """Generate image using Stability AI's API"""
    if not stability_api_key:
        st.error("Stability API key is missing. Please add your API key to the .env file.")
        return None
    
    try:
        stability_api = client.StabilityInference(
            key=stability_api_key,
            verbose=True,
            engine="stable-diffusion-xl-1024-v1-0"
        )
        
        params = st.session_state.image_generation_params
        
        # Prepare the request
        request = {
            "prompt": prompt,
            "width": params["width"],
            "height": params["height"],
            "steps": params["steps"],
            "cfg_scale": params["cfg_scale"],
            "sampler": getattr(generation, f"SAMPLER_{params['sampler']}"),
        }
        
        # Add optional parameters if they exist
        if params["style_preset"]:
            request["style_preset"] = params["style_preset"]
        if params["seed"]:
            request["seed"] = params["seed"]
        if negative_prompt:
            request["negative_prompt"] = negative_prompt
        
        answers = stability_api.generate(**request)
        
        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    st.warning("Your request activated the API's safety filters and could not be processed.")
                    return None
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img = Image.open(io.BytesIO(artifact.binary))
                    return img
        
        return None
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None

def handle_user_input(user_question):
    """Process user input and generate response with enhanced error handling and analytics"""
    if not user_question.strip():
        return
    
    # Display user message
    st.chat_message("user").write(user_question)
    
    # Handle image generation requests
    if st.session_state.model == "stable-diffusion-xl":
        with st.chat_message("assistant"):
            with st.spinner("Generating image..."):
                start_time = time.time()
                
                # Generate the image
                generated_image = generate_image(user_question)
                
                if generated_image:
                    response_time = time.time() - start_time
                    
                    # Save the image to session state
                    img_bytes = io.BytesIO()
                    generated_image.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    
                    # Store image data
                    image_data = {
                        "prompt": user_question,
                        "image": img_bytes.getvalue(),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "params": st.session_state.image_generation_params.copy()
                    }
                    st.session_state.generated_images.append(image_data)
                    
                    # Update usage stats
                    st.session_state.usage_stats["messages_sent"] += 1
                    st.session_state.usage_stats["images_generated"] += 1
                    
                    # Display the image
                    st.image(generated_image, caption=f"Generated from: '{user_question}'")
                    
                    # Save to chat history with metadata
                    message = {
                        'human': user_question, 
                        'AI': "[IMAGE GENERATED]",
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'model': st.session_state.model,
                        'response_time': response_time,
                        'image_data': image_data
                    }
                    st.session_state.chat_history.append(message)
                    
                    st.rerun()
        return
    
    # Handle text responses
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        message_container = st.container()
        
        with thinking_placeholder:
            st.info("Generating response...")
        
        try:
            start_time = time.time()
            
            # Get response from model
            response = st.session_state.conversation.invoke({'input': user_question})
            chatbot_reply = response['response']
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Calculate response speed (tokens/second)
            response_speed = calculate_response_speed(chatbot_reply, response_time)
            
            # Update usage statistics (estimated tokens)
            st.session_state.usage_stats["messages_sent"] += 1
            estimated_tokens = len(user_question.split()) + len(chatbot_reply.split())
            st.session_state.usage_stats["tokens_used"] += estimated_tokens
            
            # Track response speed for analytics
            st.session_state.usage_stats["response_speeds"].append({
                "model": st.session_state.model,
                "time": response_time,
                "speed": response_speed,
                "timestamp": datetime.now()
            })
            
            # Track model usage
            if st.session_state.model in st.session_state.usage_stats["models_used"]:
                st.session_state.usage_stats["models_used"][st.session_state.model] += 1
            else:
                st.session_state.usage_stats["models_used"][st.session_state.model] = 1
            
            # Save to chat history with metadata
            message = {
                'human': user_question, 
                'AI': chatbot_reply,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model': st.session_state.model,
                'response_time': response_time,
                'response_speed': response_speed
            }
            st.session_state.chat_history.append(message)
            
            # Remove thinking indicator and display response
            thinking_placeholder.empty()
            with message_container:
                # Apply markdown formatting to improve readability
                st.markdown(chatbot_reply)
                
                # Display response metadata in cleaner format
                st.caption(f"**{st.session_state.model}** • {response_time:.2f}s • {response_speed:.1f} tokens/sec")
            
            # Auto-scroll to bottom after new message
            st.rerun()
            
        except Exception as e:
            thinking_placeholder.empty()
            with message_container:
                error_message = str(e)
                if "api_key" in error_message.lower():
                    error_message = "API key error. Please verify your Groq API key is valid."
                elif "timeout" in error_message.lower():
                    error_message = "Request timed out. The model may be experiencing high traffic."
                
                st.error(f"⚠️ Error: {error_message}")
                st.button("Try Again", on_click=lambda: handle_user_input(user_question))

def reset_conversation():
    """Clear conversation history and reset the chat"""
    st.session_state.chat_history = []
    st.session_state.generated_images = []
    st.session_state.conversation = create_conversation(
        st.session_state.model, 
        st.session_state.memory_length,
        st.session_state.system_prompt,
        st.session_state.temperature,
        st.session_state.max_tokens
    )
    st.session_state.current_chat_title = f"Chat {len(st.session_state.saved_chats) + 1}"
    st.rerun()

def handle_settings_change():
    """Handle changes to any settings and recreate conversation"""
    st.session_state.conversation = create_conversation(
        st.session_state.model, 
        st.session_state.memory_length,
        st.session_state.system_prompt,
        st.session_state.temperature,
        st.session_state.max_tokens
    )

def save_current_chat():
    """Save current chat session"""
    if not st.session_state.chat_history:
        st.warning("Cannot save an empty chat.")
        return
        
    saved_chat = {
        "title": st.session_state.current_chat_title,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": st.session_state.model,
        "history": st.session_state.chat_history,
        "system_prompt": st.session_state.system_prompt,
        "images": st.session_state.generated_images.copy() if st.session_state.model == "stable-diffusion-xl" else []
    }
    
    st.session_state.saved_chats.append(saved_chat)
    st.success(f"Chat '{saved_chat['title']}' saved successfully!")

def load_saved_chat(chat_index):
    """Load a previously saved chat"""
    saved_chat = st.session_state.saved_chats[chat_index]
    
    st.session_state.chat_history = saved_chat["history"]
    st.session_state.model = saved_chat["model"]
    st.session_state.system_prompt = saved_chat["system_prompt"]
    st.session_state.current_chat_title = saved_chat["title"]
    
    if saved_chat["model"] == "stable-diffusion-xl" and "images" in saved_chat:
        st.session_state.generated_images = saved_chat["images"].copy()
    
    # Recreate conversation with loaded parameters
    st.session_state.conversation = create_conversation(
        st.session_state.model, 
        st.session_state.memory_length,
        st.session_state.system_prompt,
        st.session_state.temperature,
        st.session_state.max_tokens
    )
    
    st.success(f"Loaded chat: {saved_chat['title']}")
    st.rerun()

def export_chat_history(format="json"):
    """Export chat history in various formats"""
    if not st.session_state.chat_history:
        st.warning("No chat history to export.")
        return None
        
    if format == "json":
        chat_data = {
            "title": st.session_state.current_chat_title,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": st.session_state.model,
            "system_prompt": st.session_state.system_prompt,
            "messages": [{
                "role": "human" if i % 2 == 0 else "assistant",
                "content": msg["human"] if i % 2 == 0 else msg["AI"],
                "timestamp": msg.get("timestamp", ""),
                "response_time": msg.get("response_time", ""),
                "response_speed": msg.get("response_speed", "")
            } for i, msg in enumerate(st.session_state.chat_history)]
        }
        
        # Add image data if available
        if st.session_state.model == "stable-diffusion-xl" and st.session_state.generated_images:
            chat_data["images"] = [{
                "prompt": img["prompt"],
                "timestamp": img["timestamp"],
                "params": img["params"]
            } for img in st.session_state.generated_images]
        
        return json.dumps(chat_data, indent=2)
    
    elif format == "markdown":
        md_content = f"# {st.session_state.current_chat_title}\n\n"
        md_content += f"*Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        md_content += f"**Model:** {st.session_state.model}\n\n"
        md_content += f"**System Prompt:** {st.session_state.system_prompt}\n\n"
        md_content += "---\n\n"
        
        for msg in st.session_state.chat_history:
            md_content += f"## Human\n\n{msg['human']}\n\n"
            md_content += f"## Assistant\n\n{msg['AI']}\n\n"
            if 'response_time' in msg and 'response_speed' in msg:
                md_content += f"*Response time: {msg['response_time']:.2f}s • Speed: {msg['response_speed']:.1f} tokens/sec*\n\n"
            md_content += "---\n\n"
            
        # Add image section if available
        if st.session_state.model == "stable-diffusion-xl" and st.session_state.generated_images:
            md_content += "## Generated Images\n\n"
            for img in st.session_state.generated_images:
                md_content += f"### Image: {img['prompt']}\n\n"
                md_content += f"*Generated at: {img['timestamp']}*\n\n"
                md_content += "Parameters:\n```json\n"
                md_content += json.dumps(img['params'], indent=2)
                md_content += "\n```\n\n---\n\n"
            
        return md_content
    
    return None

def display_chat_analytics():
    """Display analytics about chat usage including response speed metrics"""
    st.subheader("Chat Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Messages", st.session_state.usage_stats["messages_sent"])
        st.metric("Estimated Tokens Used", st.session_state.usage_stats["tokens_used"])
        
        if st.session_state.model == "stable-diffusion-xl":
            st.metric("Images Generated", st.session_state.usage_stats["images_generated"])
        
        # Session duration
        current_time = datetime.now()
        session_duration = current_time - st.session_state.session_start_time
        hours, remainder = divmod(session_duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        st.metric("Session Duration", f"{hours}h {minutes}m {seconds}s")
        
        # Calculate average response speed
        if st.session_state.usage_stats["response_speeds"]:
            avg_speed = sum(item["speed"] for item in st.session_state.usage_stats["response_speeds"]) / len(st.session_state.usage_stats["response_speeds"])
            st.metric("Avg Response Speed", f"{avg_speed:.1f} tokens/sec")
    
    with col2:
        # Model usage chart
        if st.session_state.usage_stats["models_used"]:
            model_usage_data = pd.DataFrame({
                'Model': list(st.session_state.usage_stats["models_used"].keys()),
                'Usage Count': list(st.session_state.usage_stats["models_used"].values())
            })
            
            fig = px.pie(model_usage_data, values='Usage Count', names='Model', 
                         title='Model Usage Distribution')
            st.plotly_chart(fig, use_container_width=True)
    
    # Add speed comparison chart
    if st.session_state.usage_stats["response_speeds"]:
        st.subheader("Response Speed Analysis")
        
        # Convert the response speeds list to a DataFrame
        speeds_df = pd.DataFrame(st.session_state.usage_stats["response_speeds"])
        
        # Model speed comparison
        model_speeds = speeds_df.groupby('model')['speed'].mean().reset_index()
        fig_speed = px.bar(
            model_speeds, 
            x='model', 
            y='speed',
            title='Average Response Speed by Model (tokens/sec)',
            labels={'speed': 'Speed (tokens/sec)', 'model': 'Model'}
        )
        st.plotly_chart(fig_speed, use_container_width=True)
        
        # Speed over time
        if len(speeds_df) > 1:
            fig_time = px.line(
                speeds_df,
                x='timestamp',
                y='speed',
                color='model',
                title='Response Speed Over Time',
                labels={'speed': 'Speed (tokens/sec)', 'timestamp': 'Time'}
            )
            st.plotly_chart(fig_time, use_container_width=True)

def display_image_generation_settings():
    """Display settings for image generation"""
    st.subheader("Image Generation Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.image_generation_params["width"] = st.slider(
            "Width", 512, 2048, st.session_state.image_generation_params["width"], 64,
            help="Width of the generated image"
        )
        
        st.session_state.image_generation_params["height"] = st.slider(
            "Height", 512, 2048, st.session_state.image_generation_params["height"], 64,
            help="Height of the generated image"
        )
        
        st.session_state.image_generation_params["steps"] = st.slider(
            "Steps", 10, 150, st.session_state.image_generation_params["steps"], 5,
            help="Number of diffusion steps (more steps = higher quality but slower)"
        )
    
    with col2:
        st.session_state.image_generation_params["cfg_scale"] = st.slider(
            "CFG Scale", 1.0, 20.0, st.session_state.image_generation_params["cfg_scale"], 0.5,
            help="How closely to follow the prompt (higher = more strict)"
        )
        
        st.session_state.image_generation_params["sampler"] = st.selectbox(
            "Sampler",
            ["K_DPMPP_2M", "K_DPMPP_2S_ANCESTRAL", "K_DPM_2", "K_DPM_2_ANCESTRAL", "K_EULER", "K_EULER_ANCESTRAL"],
            index=["K_DPMPP_2M", "K_DPMPP_2S_ANCESTRAL", "K_DPM_2", "K_DPM_2_ANCESTRAL", "K_EULER", "K_EULER_ANCESTRAL"].index(
                st.session_state.image_generation_params["sampler"]
            ),
            help="Diffusion sampler method"
        )
        
        st.session_state.image_generation_params["style_preset"] = st.selectbox(
            "Style Preset (optional)",
            [None, "3d-model", "analog-film", "anime", "cinematic", "comic-book", 
             "digital-art", "enhance", "fantasy-art", "isometric", "line-art", 
             "low-poly", "modeling-compound", "neon-punk", "origami", 
             "photographic", "pixel-art", "tile-texture"],
            index=0,
            help="Predefined style to apply to the image"
        )
    
    seed_col1, seed_col2 = st.columns([3, 1])
    with seed_col1:
        seed_input = st.number_input(
            "Seed (optional)",
            min_value=0,
            max_value=2147483647,
            value=st.session_state.image_generation_params["seed"] or 0,
            help="Random seed for reproducibility (0 = random)"
        )
        st.session_state.image_generation_params["seed"] = seed_input if seed_input != 0 else None
    
    with seed_col2:
        if st.button("Random Seed", use_container_width=True):
            st.session_state.image_generation_params["seed"] = None
            st.rerun()

def main():
    # Page configuration
    st.set_page_config(
        page_title="Multimodal Chat Assistant",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS for improved UI
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 6rem;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 0.5rem;
    }
    .stSidebar .block-container {
        padding-top: 1rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .stChatMessage.user {
        background-color: rgba(240, 242, 246, 0.5);
    }
    .stChatMessage.assistant {
        background-color: rgba(240, 246, 240, 0.5);
    }
    .chat-input-container {
        position: fixed;
        bottom: 0;
        left: 25%;
        right: 0;
        background: white;
        z-index: 100;
        padding: 1rem;
        box-shadow: 0 -4px 15px rgba(0,0,0,0.08);
        border-top: 1px solid #e0e0e0;
        width: 75%;
    }
    @media (max-width: 992px) {
        .chat-input-container {
            left: 0;
            width: 100%;
        }
    }
    .chat-history {
        max-height: calc(100vh - 250px);
        overflow-y: auto;
        padding-bottom: 100px;
    }
    .stButton>button {
        border-radius: 4px;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .chat-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .scroll-to-bottom {
        max-height: 0;
        overflow-anchor: none;
    }
    .image-gallery {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    .image-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.5rem;
    }
    .image-card img {
        width: 100%;
        border-radius: 4px;
    }
    .image-prompt {
        font-size: 0.9rem;
        margin-top: 0.5rem;
        color: #555;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    initialize_session_state()
    
    # Create sidebar with tabs
    with st.sidebar:
        st.title('⚙️ Chat Settings')
        
        tabs = st.tabs(["Models", "Parameters", "Conversation", "Memory", "Saved Chats"])
        
        # Models tab
        with tabs[0]:
            st.subheader('🔍 Model Selection')
            
            # Group models by category with cleaner UI
            for category, models in MODEL_CATEGORIES.items():
                with st.expander(f"**{category} Models**", expanded=(category == "General")):
                    for model_name in models:
                        model_info = MODELS[model_name]
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{model_name}**")
                            st.caption(f"{model_info['description']}")
                        with col2:
                            if st.button(
                                "Select" if st.session_state.model != model_name else "✓ Active",
                                key=f"btn_{model_name}",
                                type="secondary" if st.session_state.model != model_name else "primary",
                                use_container_width=True
                            ):
                                st.session_state.model = model_name
                                handle_settings_change()
                                st.rerun()
            
            # Show current model details
            with st.expander("Current Model Details", expanded=True):
                current_model = MODELS[st.session_state.model]
                st.info(f"""
                **{st.session_state.model}**
                - **Category**: {current_model['category']}
                - **Token Limit**: {current_model['token_limit']}
                - **Strengths**: {current_model['strengths']}
                - **Best for**: {current_model['best_for']}
                """)
        
        # Parameters tab
        with tabs[1]:
            st.subheader('⚙️ Model Parameters')
            
            if st.session_state.model == "stable-diffusion-xl":
                display_image_generation_settings()
            else:
                # Temperature slider with cleaner UI
                temp = st.slider(
                    'Temperature:',
                    0.0, 1.0, value=st.session_state.temperature, step=0.1,
                    help="Higher values make output more random, lower values more deterministic"
                )
                if temp != st.session_state.temperature:
                    st.session_state.temperature = temp
                    handle_settings_change()
                
                # Max tokens slider
                max_tokens = st.slider(
                    'Max Output Tokens:',
                    100, 4000, value=st.session_state.max_tokens, step=100,
                    help="Maximum number of tokens in the response"
                )
                if max_tokens != st.session_state.max_tokens:
                    st.session_state.max_tokens = max_tokens
                    handle_settings_change()
        
        # Conversation tab
        with tabs[2]:
            st.subheader('💬 Conversation Mode')
            
            # Conversation modes with better styling
            conversation_mode = st.radio(
                "Select conversation mode:",
                list(SYSTEM_PROMPTS.keys()),
                index=list(SYSTEM_PROMPTS.keys()).index(st.session_state.conversation_mode),
                horizontal=True
            )
            
            if conversation_mode != st.session_state.conversation_mode:
                st.session_state.conversation_mode = conversation_mode
                st.session_state.system_prompt = SYSTEM_PROMPTS[conversation_mode]
                handle_settings_change()
            
            # Custom system prompt in an expander for cleaner UI
            with st.expander("System Prompt", expanded=False):
                custom_prompt = st.text_area(
                    "Customize system prompt:",
                    value=st.session_state.system_prompt,
                    height=150
                )
                
                if custom_prompt != st.session_state.system_prompt:
                    st.session_state.system_prompt = custom_prompt
                    handle_settings_change()
        
        # Memory tab
        with tabs[3]:
            st.subheader('🧠 Memory Settings')
            
            if st.session_state.model != "stable-diffusion-xl":
                memory_length = st.slider(
                    'Conversation memory (messages):',
                    1, 20, value=st.session_state.memory_length,
                    help="Number of previous messages to remember"
                )
                
                if memory_length != st.session_state.memory_length:
                    st.session_state.memory_length = memory_length
                    handle_settings_change()
            
            # Reset conversation button with confirmation
            with st.expander("Memory Management", expanded=True):
                if st.button("🗑️ Reset Conversation", use_container_width=True):
                    reset_conversation()
        
        # Saved Chats tab
        with tabs[4]:
            st.subheader('💾 Saved Chats')
            
            # Chat title input with better styling
            st.text_input(
                "Chat title:",
                value=st.session_state.current_chat_title,
                key="chat_title_input",
                on_change=lambda: setattr(st.session_state, 'current_chat_title', st.session_state.chat_title_input)
            )
            
            # Save current chat
            if st.button("💾 Save Current Chat", use_container_width=True):
                save_current_chat()
            
            # Load saved chats with better UI
            if st.session_state.saved_chats:
                st.divider()
                st.subheader("Your Saved Chats")
                for i, chat in enumerate(st.session_state.saved_chats):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{chat['title']}**")
                            st.caption(f"{chat['timestamp']} • {chat['model']}")
                        with col2:
                            if st.button("Load", key=f"load_{i}", use_container_width=True):
                                load_saved_chat(i)
                        st.divider()
            else:
                st.info("No saved chats yet.")
        
        # API key status indicator
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("API Status:")
        with col2:
            status = []
            if groq_api_key:
                status.append("✅ Groq")
            if stability_api_key:
                status.append("✅ Stability")
            
            if not status:
                st.caption("❌ No API keys")
            else:
                st.caption(", ".join(status))
        
        # Version information
        st.caption("Multimodal Chat Assistant v3.0")
    
    # Main chat interface with tabs
    main_tabs = st.tabs(["💬 Chat", "🖼️ Gallery", "📊 Analytics", "📤 Export"])
    
    # Chat tab
    with main_tabs[0]:
        # Chat header with info and new chat button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("Multimodal Chat Assistant 💬")
            st.caption(f"**Current Chat**: {st.session_state.current_chat_title} | **Model**: {st.session_state.model} | **Mode**: {st.session_state.conversation_mode}")
        with col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button("+ New Chat", key="new_chat_btn", use_container_width=True):
                reset_conversation()
        
        # Divider
        st.divider()
        
        # Initialize conversation if not already done
        if st.session_state.conversation is None and st.session_state.model != "stable-diffusion-xl":
            st.session_state.conversation = create_conversation(
                st.session_state.model, 
                st.session_state.memory_length,
                st.session_state.system_prompt,
                st.session_state.temperature,
                st.session_state.max_tokens
            )
        
        # Chat container with better styling
        chat_container = st.container()
        
        with chat_container:
            # Create a container for chat history with scroll
            chat_history_container = st.container()
            
            with chat_history_container:
                # Improved empty state with better styling
                if not st.session_state.chat_history:
                    st.markdown("""
                    <div style="text-align: center; padding: 80px 20px; color: #666; background-color: #f9f9f9; border-radius: 10px; margin: 20px 0;">
                        <h3>Start a New Conversation</h3>
                        <p>Select a model from the sidebar and type your message below to begin chatting.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display chat history with improved styling
                for i, message in enumerate(st.session_state.chat_history):
                    # User message
                    with st.chat_message("user"):
                        st.markdown(message['human'])
                    
                    # Assistant message with improved formatting
                    with st.chat_message("assistant"):
                        # Handle image responses
                        if 'image_data' in message:
                            st.image(message['image_data']['image'], caption=f"Generated from: '{message['human']}'")
                            st.caption(f"**{message.get('model', st.session_state.model)}** • Response time: {message.get('response_time', 0):.2f}s")
                        else:
                            # Apply better formatting for text responses
                            st.markdown(message['AI'])
                            
                            # Add metadata in a cleaner format
                            if 'response_time' in message:
                                speed_info = f"| Speed: {message.get('response_speed', 0):.1f} tokens/sec" if 'response_speed' in message else ""
                                st.caption(f"**{message.get('model', st.session_state.model)}** | Response time: {message.get('response_time', 0):.2f}s {speed_info}")
            
            # Check API key with better error message
            if not groq_api_key and st.session_state.model != "stable-diffusion-xl":
                st.error("⚠️ Groq API key is missing. Please add your API key to the .env file to enable chat functionality.")
                st.code("GROQ_API_KEY=your_api_key_here", language="bash")
                return
            
            if st.session_state.model == "stable-diffusion-xl" and not stability_api_key:
                st.error("⚠️ Stability API key is missing. Please add your API key to the .env file to enable image generation.")
                st.code("STABILITY_API_KEY=your_api_key_here", language="bash")
                return
            
            # Chat input container - only shown in the Chat tab
            with st.container():
                st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
                user_question = st.chat_input("Type your message here...", key="chat_input")
                st.markdown('</div>', unsafe_allow_html=True)
                
                if user_question:
                    handle_user_input(user_question)
    
    # Gallery tab (for image generation)
    with main_tabs[1]:
        if st.session_state.model == "stable-diffusion-xl":
            st.title("Generated Images Gallery")
            
            if st.session_state.generated_images:
                st.markdown(f"**{len(st.session_state.generated_images)} images generated in this session**")
                
                # Display images in a responsive grid
                st.markdown('<div class="image-gallery">', unsafe_allow_html=True)
                
                for img_data in reversed(st.session_state.generated_images):
                    with st.container():
                        st.markdown(f'<div class="image-card">', unsafe_allow_html=True)
                        st.image(img_data['image'], use_column_width=True)
                        st.markdown(f'<div class="image-prompt">Prompt: {img_data["prompt"]}</div>', unsafe_allow_html=True)
                        
                        # Show parameters on hover or in expander
                        with st.expander("Generation Parameters"):
                            st.json(img_data['params'])
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No images generated yet. Switch to the Chat tab and use the Stable Diffusion XL model to generate images.")
        else:
            st.info("Image gallery is only available when using the Stable Diffusion XL model. Switch models in the sidebar.")
    
    # Analytics tab
    with main_tabs[2]:
        display_chat_analytics()
    
    # Export tab
    with main_tabs[3]:
        st.title("Export Conversation")
        
        col1, col2 = st.columns(2)
        with col1:
            export_format = st.radio("Export format:", ["JSON", "Markdown"], horizontal=True)
        with col2:
            st.write("")  # Spacing
            if st.button("📥 Generate Export", use_container_width=True):
                format_key = export_format.lower()
                exported_content = export_chat_history(format_key)
                
                if exported_content:
                    # Offer download
                    st.download_button(
                        label=f"💾 Download as {export_format}",
                        data=exported_content,
                        file_name=f"{st.session_state.current_chat_title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.{format_key}",
                        mime="application/json" if format_key == "json" else "text/markdown",
                        use_container_width=True
                    )
                    
                    # Show preview in expandable section
                    with st.expander("Preview Export Content", expanded=True):
                        st.text_area("", exported_content, height=300)
                else:
                    st.warning("Nothing to export. Start a conversation first.")

if __name__ == "__main__":
    main()