Save clipboard image and describe it. Follow these steps silently:

1. Create the screenshots directory if needed and save the clipboard image:
```bash
mkdir -p ~/screenshots && timestamp=$(date +%s) && powershell.exe -command 'Add-Type -AssemblyName System.Windows.Forms; $img = [System.Windows.Forms.Clipboard]::GetImage(); if ($img) { $img.Save("C:/Users/swami.kevala/screenshots/clip_'${timestamp}'.png"); Write-Output "saved:'${timestamp}'" } else { Write-Output "no_image" }' 2>/dev/null
```

2. If the output contains "no_image", respond: "No image found in clipboard. Use Win+Shift+S to capture something first."

3. If saved successfully, read the image file at `C:/Users/swami.kevala/screenshots/clip_<timestamp>.png`

4. Respond with ONLY a brief acknowledgment of what's in the image (1-2 sentences). Do not mention the file path, the saving process, or any technical details unless the user asks.
