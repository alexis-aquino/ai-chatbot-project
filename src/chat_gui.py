import tkinter as tk
from tkinter import scrolledtext
from chatbot import chatbot_response  # ✅ full backend

# ========================
# SEND MESSAGE FUNCTION
# ========================
def send_message(event=None):
    user_input = entry_box.get().strip()
    if not user_input:
        return

    entry_box.delete(0, tk.END)

    # --- Display User message ---
    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, f"You: {user_input}\n", "user")

    try:
        bot_response = chatbot_response(user_input)
    except Exception as e:
        bot_response = f"(Error: {e})"

    # --- Display Bot message ---
    chat_window.insert(tk.END, f"Bot: {bot_response}\n\n", "bot")
    chat_window.config(state=tk.DISABLED)
    chat_window.yview(tk.END)


# ========================
# MAIN WINDOW
# ========================
root = tk.Tk()
root.title("AI Chatbot")
root.geometry("600x700")
root.resizable(True, True)  # ✅ Allow resizing
root.configure(bg="#121212")

# ========================
# CHAT WINDOW (SCROLLABLE)
# ========================
chat_window = scrolledtext.ScrolledText(
    root,
    wrap=tk.WORD,
    state=tk.DISABLED,
    bg="#1E1E1E",
    fg="#E0E0E0",
    font=("Segoe UI", 11),
    padx=12,
    pady=12,
    relief="flat",
    borderwidth=0
)
chat_window.pack(padx=15, pady=15, fill=tk.BOTH, expand=True)

# --- Tag styles for messages ---
chat_window.tag_config("user", foreground="#00ADB5", justify="right", font=("Segoe UI", 11, "bold"))
chat_window.tag_config("bot", foreground="#FFFFFF", justify="left", font=("Segoe UI", 11))

# ========================
# INPUT AREA
# ========================
input_frame = tk.Frame(root, bg="#121212")
input_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

entry_box = tk.Entry(
    input_frame,
    font=("Segoe UI", 11),
    bg="#2C2C2C",
    fg="white",
    insertbackground="white",
    relief="flat"
)
entry_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
entry_box.bind("<Return>", send_message)

send_button = tk.Button(
    input_frame,
    text="Send",
    command=send_message,
    bg="#00ADB5",
    fg="white",
    font=("Segoe UI", 10, "bold"),
    relief="flat",
    padx=15,
    pady=6,
    activebackground="#00CFC5"
)
send_button.pack(side=tk.RIGHT)

# ========================
# START GUI LOOP
# ========================
root.mainloop()
