import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
from chatbot import chatbot_response
import datetime

# ========================
# FUNCTIONS
# ========================
def send_message(event=None):
    user_input = entry_box.get().strip()
    if not user_input:
        return

    entry_box.delete(0, tk.END)
    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, f"You: {user_input}\n", "user")

    try:
        bot_response = chatbot_response(user_input)
    except Exception as e:
        bot_response = f"(Error: {e})"

    chat_window.insert(tk.END, f"Bot: {bot_response}\n\n", "bot")
    chat_window.config(state=tk.DISABLED)
    chat_window.yview(tk.END)


def clear_chat():
    chat_window.config(state=tk.NORMAL)
    chat_window.delete(1.0, tk.END)
    chat_window.config(state=tk.DISABLED)


def export_chat():
    chat_text = chat_window.get(1.0, tk.END).strip()
    if not chat_text:
        messagebox.showinfo("Export Chat", "No messages to export.")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text Files", "*.txt")],
        initialfile=f"chat_{timestamp}.txt"
    )

    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(chat_text)
        messagebox.showinfo("Export Chat", f"Chat exported to:\n{file_path}")


# ========================
# MAIN WINDOW
# ========================
root = tk.Tk()
root.title("AI Chatbot")
root.geometry("500x600")
root.configure(bg="#222831")

# --- Menu Bar ---
menubar = tk.Menu(root)
file_menu = tk.Menu(menubar, tearoff=0)
file_menu.add_command(label="Export Chat", command=export_chat)
menubar.add_cascade(label="File", menu=file_menu)

edit_menu = tk.Menu(menubar, tearoff=0)
edit_menu.add_command(label="Clear Chat", command=clear_chat)
menubar.add_cascade(label="Edit", menu=edit_menu)
root.config(menu=menubar)

# --- Chat window ---
chat_window = scrolledtext.ScrolledText(
    root, wrap=tk.WORD, state=tk.DISABLED,
    bg="#393E46", fg="white", font=("Segoe UI", 11),
    padx=10, pady=10
)
chat_window.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
chat_window.tag_config("user", foreground="#00ADB5", justify="right")
chat_window.tag_config("bot", foreground="#EEEEEE", justify="left")

# --- Input area ---
input_frame = tk.Frame(root, bg="#222831")
input_frame.pack(fill=tk.X, padx=10, pady=10)

entry_box = tk.Entry(input_frame, font=("Segoe UI", 11))
entry_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
entry_box.bind("<Return>", send_message)

send_button = tk.Button(
    input_frame, text="Send", command=send_message,
    bg="#00ADB5", fg="white", font=("Segoe UI", 10, "bold"),
    relief="flat", padx=10, pady=5
)
send_button.pack(side=tk.RIGHT)

root.mainloop()
