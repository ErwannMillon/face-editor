---
title: Face Editor
emoji: 🪞
colorFrom: yellow
colorTo: indigo
sdk: gradio
sdk_version: 3.14.0
app_file: app.py
pinned: false
---

# Face Editor
This face editor uses a CelebA pretrained VQGAN with CLIP to allow prompt-based image manipulation, as well as slider based manipulation using extracted latent vectors. 

I've written a series of Medium articles which provide a detailed and beginner-friendly explanation of how this was built. 

## Features:
Edit masking using custom backpropagation hook 


Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference