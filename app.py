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

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

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
    }
}

# Model categories for organization
MODEL_CATEGORIES = {
    "General": ["llama-3.1-8b-instant"],
    "Advanced": ["deepseek-r1-distill-llama-70b", "qwen-2.5-32b"],
    "Specialized": ["llama-3.3-70b-specdec", "qwen-2.5-coder-32b"]
}

# System prompts for different conversation modes
SYSTEM_PROMPTS = {
    "General": "You are a helpful, harmless, and honest AI assistant.",
    "Creative": "You are a creative AI assistant that helps with brainstorming, storytelling, and creative writing. Be imaginative and inspirational in your responses.",
    "Technical": "You are a technical AI assistant that specializes in programming, data analysis, and technical topics. Provide detailed and accurate technical information.",
    "Concise": "You are a concise AI assistant. Provide brief, to-the-point responses that get straight to the answer without unnecessary elaboration."
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
            "response_speeds": []
        }
    if 'saved_chats' not in st.session_state:
        st.session_state.saved_chats = []
    if 'current_chat_title' not in st.session_state:
        st.session_state.current_chat_title = f"Chat {len(st.session_state.saved_chats) + 1}"

def create_conversation(model, memory_length, system_prompt, temperature, max_tokens):
    """Create a new conversation with the specified parameters"""
    memory = ConversationBufferWindowMemory(k=memory_length)
    
    # Preload existing chat history into memory
    for message in st.session_state.chat_history:
        memory.save_context({'input': message['human']}, {'output': message['AI']})
    
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

def handle_user_input(user_question):
    """Process user input and generate response with enhanced error handling and analytics"""
    if not user_question.strip():
        return
    
    # Display user message
    st.chat_message("user").write(user_question)
    
    # Display thinking indicator
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        message_container = st.container()
        
        with thinking_placeholder:
            # Use a simpler loading indicator for better performance
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
                # Provide more user-friendly error messages
                if "api_key" in error_message.lower():
                    error_message = "API key error. Please verify your Groq API key is valid."
                elif "timeout" in error_message.lower():
                    error_message = "Request timed out. The model may be experiencing high traffic."
                
                st.error(f"⚠️ Error: {error_message}")
                st.button("Try Again", on_click=lambda: handle_user_input(user_question))

def reset_conversation():
    """Clear conversation history and reset the chat"""
    st.session_state.chat_history = []
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
        "system_prompt": st.session_state.system_prompt
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
            # Include performance metrics in export
            if 'response_time' in msg and 'response_speed' in msg:
                md_content += f"*Response time: {msg['response_time']:.2f}s • Speed: {msg['response_speed']:.1f} tokens/sec*\n\n"
            md_content += "---\n\n"
            
        return md_content
    
    return None

def display_chat_analytics():
    """Display analytics about chat usage including response speed metrics"""
    st.subheader("Chat Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Messages", st.session_state.usage_stats["messages_sent"])
        st.metric("Estimated Tokens Used", st.session_state.usage_stats["tokens_used"])
        
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

def main():
    # Page configuration
    st.set_page_config(
        page_title="Chat Assistant",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS for improved UI
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 6rem;  /* Increased bottom padding for chat input */
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
    
    /* Chat input styling */
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
    
    /* Chat history container */
    .chat-history {
        max-height: calc(100vh - 250px);
        overflow-y: auto;
        padding-bottom: 100px; /* Space for input */
    }
    
    /* Button styling */
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
    
    /* Auto-scroll to bottom */
    .scroll-to-bottom {
        max-height: 0;
        overflow-anchor: none;
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
                    reset_option = st.radio("Are you sure?", ["No", "Yes, reset conversation"])
                    if reset_option == "Yes, reset conversation":
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
            if groq_api_key:
                st.caption("✅ Connected")
            else:
                st.caption("❌ API key missing")
        
        # Version information
        st.caption("Chat Assistant v2.2")
    
    # Main chat interface with tabs
    main_tabs = st.tabs(["💬 Chat", "📊 Analytics", "📤 Export"])
    
    # Chat tab
    with main_tabs[0]:
        # Chat header with info and new chat button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("Chat Assistant 💬")
            st.caption(f"**Current Chat**: {st.session_state.current_chat_title} | **Model**: {st.session_state.model} | **Mode**: {st.session_state.conversation_mode}")
        with col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button("+ New Chat", key="new_chat_btn", use_container_width=True):
                reset_conversation()
        
        # Divider
        st.divider()
        
        # Initialize conversation if not already done
        if st.session_state.conversation is None:
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
                        # Apply better formatting for code blocks and lists
                        st.markdown(message['AI'])
                        
                        # Add metadata in a cleaner format
                        if 'response_time' in message:
                            speed_info = f"| Speed: {message.get('response_speed', 0):.1f} tokens/sec" if 'response_speed' in message else ""
                            st.caption(f"**{message.get('model', st.session_state.model)}** | Response time: {message.get('response_time', 0):.2f}s {speed_info}")
            
            # Check API key with better error message
            if not groq_api_key:
                st.error("⚠️ Groq API key is missing. Please add your API key to the .env file to enable chat functionality.")
                st.code("GROQ_API_KEY=your_api_key_here", language="bash")
                return
            
            # Chat input container - only shown in the Chat tab
            with st.container():
                st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
                user_question = st.chat_input("Type your message here...", key="chat_input")
                st.markdown('</div>', unsafe_allow_html=True)
                
                if user_question:
                    handle_user_input(user_question)
    
    # Analytics tab
    with main_tabs[1]:
        display_chat_analytics()
    
    # Export tab
    with main_tabs[2]:
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