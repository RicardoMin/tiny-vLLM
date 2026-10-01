"""HTTP entry point: python -m tinyvllm.server."""

from tinyvllm.serving.http import create_app, main


__all__ = ["create_app", "main"]


if __name__ == "__main__":
    main()