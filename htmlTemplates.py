# htmlTemplets.py

# CSS Styles
css = """
<style>
.main {
    padding: 2rem;
    background-color: #1a1a1a;
}
.stButton>button {
    width: 100%;
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    padding: 0.5rem;
    font-weight: 600;
    border: none;
    transition: all 0.3s;
}
.stButton>button:hover {
    background-color: #45a049;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.chat-message {
    padding: 1.2rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
}
.user-message {
    background-color: #ffffff;
    border-left: 4px solid #2196F3;
    color: #000000;
}
.bot-message {
    background-color: #ffffff;
    border-left: 4px solid #4CAF50;
    color: #000000;
}
h1, h2 {
    color: #ffffff;
    font-weight: 700;
}
.sidebar .sidebar-content {
    background-color: #f5f5f5;
}
.stTextInput>div>div>input {
    background-color: #ffffff;
    color: #000000;
}
.stMarkdown {
    color: #ffffff;
}
</style>
"""

# Welcome Page HTML
welcome_html = """
<div style='text-align: center; padding: 3rem;'>
    <h2>Welcome to DocuMind AI</h2>
    <p style='font-size: 1.1rem; color: #666;'>
        Upload your PDF documents in the sidebar and start asking questions.
    </p>
</div>
"""

# User Message Template
def user_message(content):
    return f"""
        <div class="chat-message user-message">
            <strong>You:</strong><br>{content}
        </div>
    """

# Bot Message Template
def bot_message(content):
    return f"""
        <div class="chat-message bot-message">
            <strong>DocuMind:</strong><br>{content}
        </div>
    """
