#!/usr/bin/env fish

source (dirname (status filename))/../lib/env.fish

while not hf download Wan-AI/Wan2.2-I2V-A14B-Diffusers --local-dir storage/models/Wan2.2-I2V-A14B-Diffusers
    echo "Download failed, retrying in 5 seconds..."
    sleep 5
end
