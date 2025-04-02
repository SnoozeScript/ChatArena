# Chat Assistant

## 📌 About
Chat Assistant is an AI-powered chatbot built using [Groq](https://console.groq.com) and LangChain. It provides an interactive chat experience through Streamlit, allowing users to select AI models, manage conversation memory, and maintain chat history.



## 🔑 How to Get a Groq API Key
To use the Groq AI models, follow these steps:
1. **Visit** [Groq Console](https://console.groq.com).
2. **Sign In** or **Create an Account**.
3. **Navigate to API Keys** in the console.
4. **Generate a New API Key**.
5. **Copy the Key** and store it safely.
6. **Add the Key** to a `.env` file in your project:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```

## 🛠️ Installation
### Prerequisites
- Python 3.8 or later installed.
- Pip installed.

### Steps to Install
Clone the repository and install the required dependencies:
```sh
git clone https://github.com/your-repo/chat-assistant.git
cd chat-assistant
pip install -r requirements.txt
```

## ▶️ Running the Application
Ensure your `.env` file is correctly set up, then run the following command:
```sh
streamlit run app.py
```
This will start the chatbot on your local machine.

## 📂 Project Structure
```
├── app.py                # Main application script
├── requirements.txt      # Dependencies
├── .env                  # API Key Storage
├── README.md             # Documentation
```

## 🤖 Supported AI Models
| Model Name                        | Description |
|------------------------------------|-------------|
| `llama-3.1-8b-instant`            | Fast and efficient AI model |
| `deepseek-r1-distill-qwen-32b`    | Advanced model with higher accuracy |

## 📝 Usage Guide
1. **Start the app** using `streamlit run app.py`.
2. **Select AI Model** from the sidebar.
3. **Set memory length** to decide how many messages the AI remembers.
4. **Chat with the AI** by entering your queries.
5. **Monitor API Status** in the sidebar.

## 🛡️ API Key Status
The sidebar will indicate whether the API key is connected. If not, check your `.env` file to ensure the key is correctly set.


