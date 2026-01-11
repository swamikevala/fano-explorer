Save clipboard image and describe it. Follow these steps silently:

1. Save the clipboard image:

```bash
.claude/scripts/clipboard-save.sh
```

2. If the output contains "no_image", respond: "No image found in clipboard. Use Win+Shift+S to capture something first."
3. If saved successfully, read the image file at `~/screenshots/clip_<timestamp>.png` (use the timestamp from the output)
4. Respond with ONLY a brief acknowledgment of what's in the image (1-2 sentences). Do not mention the file path, the saving process, or any technical details unless the user asks.
