import os
import json
import time
import uuid
import requests
import base64
import re
import argparse
from dotenv import load_dotenv

# Using the brand new, officially supported Google GenAI SDK
from google import genai
from google.genai import types
from google.genai.errors import APIError

from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips, CompositeAudioClip
import moviepy.video.fx.all as vfx
import moviepy.audio.fx.all as afx

# ==============================================================================
# CONFIGURATION & ENVIRONMENT SETUP
# ==============================================================================

# 🛑 DEVELOPMENT MODE TOGGLE 🛑
DEV_MODE = True

# ⚡ RENDER SPEED TOGGLE ("test" or "production") ⚡
RENDER_QUALITY = "test"

# Load environment variables from the local .env file
load_dotenv()

# ==============================================================================
# WINDOWS IMAGEMAGICK FIX
# ==============================================================================
from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"})

# ==============================================================================
# MODEL ROUTING & API KEY CYCLING CONFIGURATION
# ==============================================================================
ROUTING_LOGIC = {
    "heavy_reasoning": ["gemini-1.5-pro","gemini-3.5-flash","gemini-2.5-flash","gemini-3.1-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash"]
}

# Load keys as lists. Use commas in your .env file
GEMINI_KEYS = [k.strip() for k in os.environ.get("GEMINI_API_KEYS", os.environ.get("GEMINI_API_KEY", "")).split(",") if k.strip()]
ELEVEN_KEYS = [k.strip() for k in os.environ.get("ELEVENLABS_API_KEYS", os.environ.get("ELEVENLABS_API_KEY", "")).split(",") if k.strip()]
PEXELS_KEYS = [k.strip() for k in os.environ.get("PEXELS_API_KEYS", os.environ.get("PEXELS_API_KEY", "")).split(",") if k.strip()]

if not GEMINI_KEYS or not ELEVEN_KEYS or not PEXELS_KEYS:
    print("❌ API Key Error: Missing one or more API key lists. Please check your .env file.")
    exit(1)

current_gemini_idx = 0
current_eleven_idx = 0
current_pexels_idx = 0

# ==============================================================================
# PROFILE MANAGEMENT (PRODUCTION VIRAL METADATA)
# ==============================================================================

def load_or_create_profile(profile_name):
    os.makedirs("profiles", exist_ok=True)
    profile_path = os.path.join("profiles", f"{profile_name}.json")
    
    if os.path.exists(profile_path):
        with open(profile_path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    print(f"⚠️ Profile '{profile_name}' not found. Auto-generating expanded Urban Mysteries profile...")
    
    default_profile = {
        "theme_name": "Urban Mysteries",
        "voice_id": "TX3LPaxmHKxFdv7VOQHJ", # Liam
        "voice_model": "eleven_multilingual_v2",
        "voice_stability": 0.30,
        "voice_style": 0.65,
        "bg_music_file": "bg_music.mp3",
        "bg_music_volume": 0.08,
        "visual_settings": {
            "font_family": "Arial-Black",
            "font_color": "#FFFF00",
            "stroke_color": "black",
            "caption_y_percentage": 0.60
        },
        "system_prompt": """
You are a viral YouTube Shorts, TikTok, and Instagram Reels producer specializing in high-retention storytelling. 
Create a unique 45-second script based on shocking urban anomalies, financial system glitches, or bizarre real-world modern mysteries.

Do NOT use any of these past topics: {history}

CRITICAL SCRIPT FORMATTING RULE (FOR AI VOICE PACING):
You MUST format the "script" text to sound natural but keep it clean:
1. Use simple punctuation like commas (,) and periods (.) strategically to force micro-pauses and natural breathing.
2. Do NOT use em-dashes (—) or quotation marks (" "). Keep the punctuation basic.
3. Write out dates and numbers using digits (e.g., "2026" or "100").

CRITICAL VISUAL B-ROLL RULE:
The 'tags' array MUST contain exactly 16 simple, 1-2 word NOUNS that match the chronological story beats for Pexels. Use generic searchable nouns.

🚀 CRITICAL VIRAL SOCIAL METADATA ENGINE:
You must also generate hyper-optimized viral metadata customized for platform APIs. 
Fill in the text fields dynamically based on the story, but KEEP the boolean/integer fields exactly as they appear in the structure below.

Output ONLY a JSON object with this exact structure:
{
  "title": "Internal working title",
  "script": "The pacing-optimized spoken script. Around 110-130 words.",
  "tags": ["noun 1", "noun 2", "noun 3", "noun 4", "noun 5", "noun 6", "noun 7", "noun 8", "noun 9", "noun 10", "noun 11", "noun 12", "noun 13", "noun 14", "noun 15", "noun 16"],
  "metadata": {
    "youtube_shorts": {
      "title": "Curiosity-gap hook under 60 chars",
      "description": "Engaging 2-sentence breakdown ending with trending shorts tags.",
      "tags": ["urbanlegends", "mysteries", "shorts", "finance"],
      "made_for_kids": false,
      "category_id": "24"
    },
    "tiktok": {
      "caption": "High-converting retention question + niche tags like #fyp #mystery",
      "disable_comment": false,
      "disable_duet": false,
      "disable_stitch": false,
      "video_cover_timestamp_ms": 1500,
      "brand_content_toggle": false
    },
    "instagram_reels": {
      "caption": "High-value storytelling layout ending with 'Follow for more daily mysteries.' #reelsviral #mysterychannel",
      "share_to_feed": true,
      "cover_image_timestamp_ms": 1500
    }
  }
}
"""
    }
    
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(default_profile, f, indent=4)
        
    return default_profile

# ==============================================================================
# WATERFALL ROUTER
# ==============================================================================

def generate_with_fallback(contents, model_queue, config=None):
    global current_gemini_idx
    while current_gemini_idx < len(GEMINI_KEYS):
        current_key = GEMINI_KEYS[current_gemini_idx]
        temp_client = genai.Client(api_key=current_key)
        
        for i, model_name in enumerate(model_queue):
            try:
                response = temp_client.models.generate_content(
                    model=model_name, contents=contents, config=config
                )
                return response
            except APIError as e:
                err_str = str(e).lower()
                if any(k in err_str for k in ["429", "503", "500", "404", "not_found", "quota", "exhausted"]):
                    time.sleep(2) 
                    continue
                raise e
        current_gemini_idx += 1
    raise Exception("Gemini API limits reached on all keys.")

# ==============================================================================
# CORE WORKFLOW FUNCTIONS
# ==============================================================================

def load_history(history_file):
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(history_file, topic):
    history = load_history(history_file)
    history.append(topic)
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

def generate_topic_script_tags(profile, history, script_cache_file, is_batching):
    if DEV_MODE and not is_batching and os.path.exists(script_cache_file):
        with open(script_cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    print("🧠 Brainstorming unique script and generating viral properties...")
    prompt = profile["system_prompt"].replace("{history}", json.dumps(history))
    
    response = generate_with_fallback(
        contents=prompt,
        model_queue=ROUTING_LOGIC["heavy_reasoning"],
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    
    data = json.loads(response.text)
    print(f"🎯 Topic Selected: {data['title']}")
    
    with open(script_cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        
    return data

def generate_audio_and_captions(script_text, profile, audio_path, timestamps_cache_file, is_batching):
    global current_eleven_idx
    
    if DEV_MODE and not is_batching and os.path.exists(audio_path) and os.path.exists(timestamps_cache_file):
        with open(timestamps_cache_file, "r", encoding="utf-8") as f:
            return audio_path, json.load(f)

    print("🎙️ Generating ElevenLabs voiceover...")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{profile['voice_id']}/with-timestamps"
    data = {
        "text": script_text,
        "model_id": profile.get("voice_model", "eleven_multilingual_v2"),
        "voice_settings": {
            "stability": profile.get("voice_stability", 0.30), 
            "similarity_boost": 0.80, 
            "style": profile.get("voice_style", 0.65),
            "use_speaker_boost": True
        }
    }
    
    while current_eleven_idx < len(ELEVEN_KEYS):
        headers = {"Content-Type": "application/json", "xi-api-key": ELEVEN_KEYS[current_eleven_idx]}
        resp = requests.post(url, json=data, headers=headers)
        
        if resp.status_code in [401, 402, 429]:
            current_eleven_idx += 1
            time.sleep(1)
            continue
        resp.raise_for_status()
        
        response_data = resp.json()
        with open(audio_path, 'wb') as f:
            f.write(base64.b64decode(response_data["audio_base64"]))
            
        chars = response_data["alignment"]["characters"]
        starts = response_data["alignment"]["character_start_times_seconds"]
        ends = response_data["alignment"]["character_end_times_seconds"]

        words, current_word, word_start = [], "", None
        for i, char in enumerate(chars):
            if char == " ":
                if current_word:
                    words.append((word_start, ends[i-1], current_word))
                    current_word, word_start = "", None
            else:
                if current_word == "": word_start = starts[i]
                current_word += char
        if current_word: words.append((word_start, ends[-1], current_word))

        subs = []
        for w_start, w_end, word in words:
            clean_word = re.sub(r'[,—\-"“”\.]', '', word).strip()
            if clean_word: subs.append([w_start, w_end, clean_word])

        with open(timestamps_cache_file, "w", encoding="utf-8") as f:
            json.dump(subs, f, indent=4)
        return audio_path, subs

    raise Exception("❌ CRITICAL: All ElevenLabs keys are exhausted.")

def get_pexels_data(url):
    global current_pexels_idx
    while current_pexels_idx < len(PEXELS_KEYS):
        headers = {"Authorization": PEXELS_KEYS[current_pexels_idx]}
        resp = requests.get(url, headers=headers)
        if resp.status_code == 429:
            current_pexels_idx += 1
            time.sleep(2)
            continue
        resp.raise_for_status()
        return resp.json()
    raise Exception("❌ Pexels API exhausted.")

def download_b_roll(tags, assets_dir, is_batching): 
    required = 16
    existing = [os.path.join(assets_dir, f) for f in os.listdir(assets_dir) if f.startswith("broll_") and f.endswith(".mp4")]
    
    if DEV_MODE and not is_batching and len(existing) >= required:
        # Sort existing to ensure they stay in the expected chronological order
        return sorted(existing)[:required]

    print(f"🎬 Sourcing strictly ONE video per tag slot...")
    downloaded = []
    
    for i, tag in enumerate(tags[:required]):
        search_queries = [tag, tag.split()[0], "cinematic abstract dark"]
        found = False
        
        # Create a filesystem-safe version of the tag
        safe_tag = "".join([c for c in tag if c.isalnum() or c == ' ']).strip().replace(' ', '_')
        if not safe_tag:
            safe_tag = "fallback"
        
        for query in search_queries:
            if found: break 
            url = f"https://api.pexels.com/videos/search?query={query}&orientation=portrait&per_page=5"
            try:
                data = get_pexels_data(url)
                for video in data.get('videos', []):
                    if video.get('duration', 0) < 5: continue 
                    files = video.get('video_files', [])
                    hd_files = [v for v in files if v.get('quality') == 'hd' and v.get('width', 0) >= 720]
                    best_file = hd_files[0] if hd_files else files[0]
                    
                    vid_resp = requests.get(best_file['link'])
                    
                    # Inject chronological index and the tag directly into the filename
                    filename = os.path.join(assets_dir, f"broll_{i:02d}_{safe_tag}_{uuid.uuid4().hex[:4]}.mp4")
                    
                    with open(filename, 'wb') as f: f.write(vid_resp.content)
                    downloaded.append(filename)
                    found = True
                    break 
            except Exception:
                pass
    return downloaded

def assemble_video(audio_path, bg_music_path, valid_videos, subs_data, final_title, output_dir, profile):
    print("🎞️ Stitching visual, audio, and captions in MoviePy...")
    audio = AudioFileClip(audio_path)
    audio_duration = audio.duration
    clip_duration = audio_duration / len(valid_videos)

    if RENDER_QUALITY == "test":
        target_w, target_h, render_fps, font_size, stroke_thickness, offset_shadow = 540, 960, 30, 50, 7, 4
    else:
        target_w, target_h, render_fps, font_size, stroke_thickness, offset_shadow = 1080, 1920, 60, 98, 15, 8

    # Extract dynamic settings from the profile
    bg_vol = profile.get("bg_music_volume", 0.08)
    vis_settings = profile.get("visual_settings", {})
    font_choice = vis_settings.get("font_family", "Arial-Black")
    font_color = vis_settings.get("font_color", "#FFFF00")
    stroke_col = vis_settings.get("stroke_color", "black")
    y_perc = vis_settings.get("caption_y_percentage", 0.60)

    clips = []
    
    for vid in valid_videos:
        clip = VideoFileClip(vid).without_audio().set_fps(render_fps)
        w, h = clip.size
        
        # 🚀 ANTI-BLACK-BAR FIX: Calculate the scaling factor needed to fill both dimensions
        scale_w = target_w / float(w)
        scale_h = target_h / float(h)
        scale_factor = max(scale_w, scale_h) # Always scale by the larger requirement to guarantee full coverage
        
        # Resize the clip using the master scale factor
        clip = clip.resize(scale_factor)
        
        # Now that the clip is guaranteed to be equal to or larger than the target on both axes, crop the exact center
        clip = clip.crop(x_center=clip.w/2.0, y_center=clip.h/2.0, width=target_w, height=target_h)
            
        if clip.duration < clip_duration:
            clip = clip.fx(vfx.loop, duration=clip_duration)
        else:
            start_time = 2.0 if clip.duration >= (clip_duration + 2.0) else 0.0
            clip = clip.subclip(start_time, start_time + clip_duration)
            
        clips.append(clip)
        
    final_visual = concatenate_videoclips(clips, method="compose")
    
    bg_clip = None
    if bg_music_path and os.path.exists(bg_music_path):
        bg_clip = AudioFileClip(bg_music_path).fx(afx.volumex, bg_vol)
        bg_clip = afx.audio_loop(bg_clip, duration=audio_duration)
        final_audio = CompositeAudioClip([audio, bg_clip])
    else:
        final_audio = audio
        
    final_visual = final_visual.set_audio(final_audio).subclip(0, audio_duration)
    text_clips = []
    
    def snappy_pop(t):
        if t < 0.075: return 0.85 + 2.66 * t  
        elif t < 0.15: return 1.05 - 0.66 * (t - 0.075) 
        return 1.0

    caption_y_pos = int(target_h * y_perc)

    for start, end, text in subs_data:
        if end <= start or start > audio_duration: continue
        end = min(end, audio_duration)
        
        raw_text = text.upper()
        tight_kerning = -5 if RENDER_QUALITY != "test" else -2
        
        txt_shadow = TextClip(raw_text, fontsize=font_size, color=stroke_col, font=font_choice, stroke_color=stroke_col, stroke_width=stroke_thickness, kerning=tight_kerning, method='label', align='center')
        txt_stroke = TextClip(raw_text, fontsize=font_size, color=stroke_col, font=font_choice, stroke_color=stroke_col, stroke_width=stroke_thickness, kerning=tight_kerning, method='label', align='center')
        txt_fill = TextClip(raw_text, fontsize=font_size, color=font_color, font=font_choice, stroke_width=0, kerning=tight_kerning, method='label', align='center')
        
        box_w, box_h = max(txt_shadow.w, txt_stroke.w, txt_fill.w) + 40, max(txt_shadow.h, txt_stroke.h, txt_fill.h) + 40
        cx, cy = (box_w - txt_stroke.w) / 2, (box_h - txt_stroke.h) / 2
        
        txt_shadow = txt_shadow.set_position((cx + offset_shadow, cy + offset_shadow))
        txt_stroke = txt_stroke.set_position((cx, cy))
        txt_fill = txt_fill.set_position(((box_w - txt_fill.w) / 2, (box_h - txt_fill.h) / 2))
        
        word_clip = CompositeVideoClip([txt_shadow, txt_stroke, txt_fill], size=(box_w, box_h)).resize(snappy_pop).set_position(('center', caption_y_pos)).set_start(start).set_end(end)
        text_clips.append(word_clip)
        
    final_video = CompositeVideoClip([final_visual] + text_clips)
    safe_title = "".join([c for c in final_title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
    base_filename = safe_title.replace(' ', '_')
    output_path = os.path.join(output_dir, f"{base_filename}.mp4")
    
    final_video.write_videofile(output_path, fps=render_fps, codec="libx264", audio_codec="aac", preset="ultrafast", threads=4, logger='bar')
    
    final_video.close()
    final_visual.close()
    audio.close()
    if bg_clip: bg_clip.close()
    for clip in clips: clip.close()

    print("🎉 Render complete and resources released.")
    return output_path, base_filename

# ==============================================================================
# SURGICAL CLEANUP
# ==============================================================================

def cleanup_workspace(assets_dir, bg_music_filename):
    print("🧹 Running surgical cleanup (Leaving history and bg_music intact)...")
    for file in os.listdir(assets_dir):
        if file.endswith(".mp4") or file.endswith(".jpg") or file == "voiceover.mp3" or "cache" in file:
            if file != bg_music_filename:
                try: os.remove(os.path.join(assets_dir, file))
                except Exception: pass

# ==============================================================================
# MAIN BATCH PIPELINE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Batch Video Slop Generator")
    parser.add_argument("--profile", type=str, default="urban_mysteries", help="Name of the channel profile to load")
    parser.add_argument("--count", type=int, default=1, help="Number of videos to generate in this batch")
    args = parser.parse_args()

    print(f"🔥 INITIALIZING BATCH PIPELINE: Profile [{args.profile}] | Target Count: {args.count} 🔥")

    profile = load_or_create_profile(args.profile)
    
    assets_dir = f"assets_{args.profile}"
    base_output_dir = f"output_{args.profile}"
    history_file = os.path.join(assets_dir, "history.json")
    script_cache = os.path.join(assets_dir, "script_cache.json")
    timestamps_cache = os.path.join(assets_dir, "timestamps_cache.json")
    local_bg_music = os.path.join(assets_dir, profile["bg_music_file"])
    
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(base_output_dir, exist_ok=True)

    # LOOP BATCH ENGINE
    for i in range(args.count):
        print(f"\n=======================================================")
        print(f"🎬 STARTING VIDEO {i+1} OF {args.count}")
        print(f"=======================================================")
        
        is_batching = args.count > 1 
        history = load_history(history_file)
        
        gemini_data = generate_topic_script_tags(profile, history, script_cache, is_batching)
        title = gemini_data['title']
        
        safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
        base_filename = safe_title.replace(' ', '_')
        video_out_dir = os.path.join(base_output_dir, base_filename)
        os.makedirs(video_out_dir, exist_ok=True)
        
        audio_path, subs_data = generate_audio_and_captions(
            gemini_data['script'], profile, os.path.join(assets_dir, "voiceover.mp3"), timestamps_cache, is_batching
        )
        
        bg_music_path = local_bg_music if os.path.exists(local_bg_music) else None
        valid_videos = download_b_roll(gemini_data['tags'], assets_dir, is_batching)
        
        # 🚀 Pass the entire profile down to assemble_video
        _, returned_base_filename = assemble_video(audio_path, bg_music_path, valid_videos, subs_data, title, video_out_dir, profile)
        
        # 🚀 WRITE COMPANION METADATA JSON ARTIFACT
        if 'metadata' in gemini_data:
            metadata_output_path = os.path.join(video_out_dir, f"{returned_base_filename}.json")
            with open(metadata_output_path, "w", encoding="utf-8") as meta_f:
                json.dump(gemini_data['metadata'], meta_f, indent=4)
            print(f"📦 Paired viral properties saved to: {metadata_output_path}")
        
        save_history(history_file, title)
        
        if i < args.count - 1 or not DEV_MODE:
            cleanup_workspace(assets_dir, profile["bg_music_file"])

    print(f"\n🎉 BATCH COMPLETE! Generated {args.count} videos and metadata packets in '{base_output_dir}'.")

if __name__ == "__main__":
    main()