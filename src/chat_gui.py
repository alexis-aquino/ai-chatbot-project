import tkinter as tk
from tkinter import scrolledtext

# ✅ Import chatbot_response instead of get_response
from chatbot import chatbot_response  


def send_message():
    user_input = entry_box.get().strip()
    if not user_input:
        return

    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, "You: " + user_input + "\n")

    try:
        # ✅ Use the full chatbot pipeline (deep model + NER + API)
        bot_response = chatbot_response(user_input)
    except Exception as e:
        bot_response = f"(Error: {e})"

    chat_window.insert(tk.END, "Bot: " + bot_response + "\n\n")
    chat_window.config(state=tk.DISABLED)
    chat_window.yview(tk.END)

    entry_box.delete(0, tk.END)


# --- Main window ---
root = tk.Tk()
root.title("AI Chatbot")
root.geometry("500x600")
root.resizable(False, False)

# --- Chat window ---
chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED)
chat_window.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# --- Entry box ---
entry_box = tk.Entry(root, width=60)
entry_box.pack(padx=10, pady=10, side=tk.LEFT, fill=tk.X, expand=True)

# Bind Enter key
entry_box.bind("<Return>", lambda event: send_message())

# --- Send button ---
send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack(padx=10, pady=10, side=tk.RIGHT)

root.mainloop()
