from __future__ import annotations

import os
import sys


def chat_loop(run_turn, *, system="", title="tiny-vLLM", errors=(ValueError,)) -> None:
    """Shared terminal UI; run_turn(messages, on_text) returns a complete reply."""
    color = (
        sys.stdin.isatty()
        and sys.stdout.isatty()
        and "NO_COLOR" not in os.environ
        and os.environ.get("TERM") != "dumb"
    )

    def label(text, code):
        return f"\033[{code}m{text}\033[0m" if color else text 
    
    help_text = (
        "/help  commands   /reset  clear history   /exit  quit\n"
        "Ctrl-C: cancel generation; at the input prompt, exit."
    )
    initial = [{"role": "system", "content": system}] if system else []
    history = initial.copy()
    print(f"\n{label(title, '1;36')}\n{help_text}\n")
    
    while True:
        try:
            prompt = input(label("You > ", "1;32")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not prompt:
            continue
        if prompt == "/exit":
            return
        if prompt == "/help":
            print(f"{help_text}\n")
            continue
        if prompt == "/reset":
            history = initial.copy()
            print("History cleared; system prompt retained.\n")
            continue
        
        messages = [*history, {"role": "user", "content": prompt}]
        print(label("\nAssistant > ", "1;36"), end="", flush=True)
        try:
            text = run_turn(
                messages,
                lambda text: print(text, end="", flush=True),
            )
        except KeyboardInterrupt:
            print("\n[cancelled; history unchanged]", file=sys.stderr)
        except errors as error:
            print(f"\n[failed: {error}; history unchanged]", file=sys.stderr)
        else:
            history = [*messages, {"role": "assistant", "content": text}]
        print("\n", flush=True)