css = """
<style>

/* Overall chat container */
.chat-container {
    max-width: 900px;
    margin: auto;
}

/* Message bubble */
.chat-message {
    padding: 1rem 1.2rem;
    border-radius: 14px;
    margin-bottom: 1rem;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    animation: fadeIn 0.3s ease-in-out;
}

/* User message */
.chat-message.user {
    background: linear-gradient(135deg, #1f2937, #111827);
    border-left: 4px solid #3b82f6;
}

/* Bot message */
.chat-message.bot {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border-left: 4px solid #22c55e;
}

/* Avatar container */
.chat-message .avatar {
    width: 48px;
    height: 48px;
    flex-shrink: 0;
}

/* Avatar image */
.chat-message .avatar img {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid rgba(255, 255, 255, 0.2);
}

/* Message text */
.chat-message .message {
    color: #e5e7eb;
    line-height: 1.6;
    font-size: 0.95rem;
    word-wrap: break-word;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid #1e293b;
}

/* Fade animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

</style>
"""


bot_template = """
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png">
    </div>
    <div class="message">
        {{MSG}}
    </div>
</div>
"""

user_template = """
<div class="chat-message user">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/9131/9131529.png">
    </div>
    <div class="message">
        {{MSG}}
    </div>
</div>
"""
