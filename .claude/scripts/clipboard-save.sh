#!/bin/bash
mkdir -p ~/screenshots
timestamp=$(date +%s)
powershell.exe -command "Add-Type -AssemblyName System.Windows.Forms; \$img = [System.Windows.Forms.Clipboard]::GetImage(); if (\$img) { \$img.Save('C:/Users/swami.kevala/screenshots/clip_${timestamp}.png'); Write-Output 'saved:${timestamp}' } else { Write-Output 'no_image' }"