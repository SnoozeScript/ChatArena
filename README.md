
# Chat Assistant

## 📌 About  
Chat Assistant is an AI-powered chatbot built using [Groq](https://console.groq.com) and LangChain. It provides an interactive chat experience through Streamlit, allowing users to select AI models, manage conversation memory, and maintain chat history.

---

## 🔑 How to Get a Groq API Key  
To use the Groq AI models, follow these steps:

1. **Visit** [Groq Console](https://console.groq.com)  
2. **Sign In** or **Create an Account**  
3. **Navigate to API Keys** in the console  
4. **Generate a New API Key**  
5. **Copy the Key** and store it safely  
6. **Add the Key** to a `.env` file in your project:
   ```env
   GROQ_API_KEY=your_api_key_here
   STABILITY_API_KEY=your_api_key_here
   ```

---

## 🛠️ Installation  

### Prerequisites  
- Python 3.8 or later installed  
- Pip installed  

### Steps to Install  
Clone the repository and install the required dependencies:
```sh
git clone https://github.com/your-repo/chat-assistant.git
cd chat-assistant
pip install -r requirements.txt
```

---

## ▶️ Running the Application  
Ensure your `.env` file is correctly set up, then run the following command:
```sh
streamlit run app.py
```
This will start the chatbot on your local machine.

---

## 📂 Project Structure
```
├── app.py                # Main application script  
├── requirements.txt      # Dependencies  
├── .env                  # API Key Storage  
├── README.md             # Documentation  
```

---

## 🤖 Supported AI Models  

| Icon | Model Name                       | Description                                               | Category       | Token Limit | Best For                                   |
|------|----------------------------------|-----------------------------------------------------------|----------------|-------------|---------------------------------------------|
| ⚡   | `llama-3.1-8b-instant`           | Fast, efficient model for quick responses                | General        | 16,384      | Quick conversations, basic tasks           |
| 🧠   | `deepseek-r1-distill-llama-70b`  | Advanced distilled model with excellent performance       | Advanced       | 32,768      | Complex reasoning, detailed explanations   |
| 🎯   | `qwen-2.5-32b`                   | High-quality model for detailed responses                 | Advanced       | 32,768      | Longer conversations, nuanced responses    |
| 🚀   | `llama-3.3-70b-specdec`         | Speculative decoding-enhanced Llama 3 for faster responses| Specialized    | 16,384      | Fast English generation, coding assistance |
| 💻   | `qwen-2.5-coder-32b`            | Model optimized for coding tasks                          | Specialized    | 32,768      | Programming assistance, technical docs     |
| 🎨   | `stable-diffusion-xl`           | Stability AI's advanced image generation model            | Image Generation| N/A        | Generating images from text prompts        |

---

## 📝 Usage Guide  

1. **Start the app** using `streamlit run app.py`  
2. **Select AI Model** from the sidebar  
3. **Set memory length** to decide how many messages the AI remembers  
4. **Chat with the AI** by entering your queries  
5. **Monitor API Status** in the sidebar  

---

## 🛡️ API Key Status  
The sidebar will indicate whether the API key is connected.  
If not, check your `.env` file to ensure the key is correctly set.
