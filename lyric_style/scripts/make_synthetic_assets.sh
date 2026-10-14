#!/usr/bin/env bash
# Generate SELF-AUTHORED test assets (no third-party content) so the whole
# pipeline can be exercised end to end. Everything produced here is owned by
# you (licence "self-authored") and is registered in the licence manifest.
#
#   bash scripts/make_synthetic_assets.sh
#
# Footage: abstract moving gradients (no real faces — the planner handles
#          face_count==0). Music: a 125 BPM kick+click bed, 32 s.
set -eu
cd "$(dirname "$0")/.."
mkdir -p assets/footage assets/music

FONT=assets/fonts/Anton-Regular.ttf

for i in 1 2 3 4 5 6; do
  hue=$((i * 40))
  out="assets/footage/synth_${i}.mp4"
  # A drifting gradient with a fine texture grid + a moving "face" block, so
  # sharpness/settle metrics are meaningful. 12 s, 1280x720, 30 fps.
  ffmpeg -y -v error \
    -f lavfi -i "gradients=s=1280x720:d=12:speed=0.05:c0=0x$(printf '%02x' $((hue)) )3355:c1=0x112244:type=linear" \
    -filter_complex "[0:v]hue=h=${hue},eq=brightness=0.02,\
noise=alls=18:allf=t+u[bg];\
[bg]drawgrid=w=16:h=16:t=1:c=white@0.10,\
drawbox=x='100+80*sin(t)':y=180:w=380:h=420:color=0xE0C0A0@0.9:t=fill,\
drawbox=x='120+80*sin(t)':y=230:w=60:h=22:color=0x101010@0.95:t=fill,\
drawbox=x='400+80*sin(t)':y=230:w=60:h=22:color=0x101010@0.95:t=fill,\
drawbox=x='180+80*sin(t)':y=430:w=220:h=28:color=0x902020@0.85:t=fill,\
drawbox=x='250+80*sin(t)':y=350:w=80:h=40:color=0xC08080@0.9:t=fill[v]" \
    -map "[v]" -t 12 -r 30 -c:v libx264 -preset veryfast -crf 20 \
    -pix_fmt yuv420p "$out"
  echo "wrote $out"
done

# 125 BPM => bar = 1.92 s (matches the reference feel).
ffmpeg -y -v error -f lavfi \
  -i "aevalsrc=0.9*sin(2*PI*60*t)*exp(-22*mod(t\,0.48))+0.35*sin(2*PI*3000*t)*exp(-60*mod(t\,0.48)):s=48000:d=32:c=stereo" \
  -af "loudnorm=I=-16:TP=-1.5:LRA=7" -c:a pcm_s16le assets/music/synth_125bpm.wav
echo "wrote assets/music/synth_125bpm.wav"
