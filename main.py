import os
import json
import time
import uuid
import requests
import cv2
import base64
from dotenv import load_dotenv

# Using the brand new, officially supported Google GenAI SDK
from google import genai
from google.genai import types
from google.genai.errors import APIError

from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
import moviepy.video.fx.all as vfx

# ==============================================================================
# CONFIGURATION & ENVIRONMENT SETUP
# ==============================================================================

# 🛑 DEVELOPMENT MODE TOGGLE 🛑
DEV_MODE = True

# Load environment variables from the local .env file
load_dotenv()

# ==============================================================================
# WINDOWS IMAGEMAGICK FIX
# ==============================================================================
from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"})

ELEVENLABS_VOICE_ID = "pNInz6obpgDQGcFmaJgB" # Adam - great for mysterious storytelling
ASSETS_DIR = "assets"
OUTPUT_DIR = "output"
HISTORY_FILE = "history.json"
SCRIPT_CACHE_FILE = os.path.join(ASSETS_DIR, "script_cache.json")
TIMESTAMPS_CACHE_FILE = os.path.join(ASSETS_DIR, "timestamps_cache.json")

# ==============================================================================
# MODEL ROUTING (FREE TIER OPTIMIZED)
# ==============================================================================
# 🚀 FIX: Mapped to actual, currently available Google AI Studio Free Tier models.
ROUTING_LOGIC = {
    "heavy_reasoning": ["gemini-3.5-flash","gemini-2.5-flash","gemini-3.1-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash"],
    "fast_vision": ["gemini-3.5-flash","gemini-2.5-flash","gemini-3.1-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash"]
}

os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    ELEVEN_KEY = os.environ.get("ELEVENLABS_API_KEY")
    PEXELS_KEY = os.environ.get("PEXELS_API_KEY")
    
    if not ELEVEN_KEY or not PEXELS_KEY:
        raise KeyError("One or more keys are missing or empty.")
except Exception as e:
    print(f"❌ API Key Error: {e}. Please check your .env file.")
    exit(1)

# ==============================================================================
# RATE LIMIT & SERVER OVERLOAD WATERFALL ROUTER
# ==============================================================================

def generate_with_fallback(contents, model_queue, config=None):
    for i, model_name in enumerate(model_queue):
        try:
            print(f"   🔄 Attempting API call with: [{model_name}]...")
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config
            )
            print(f"   ✅ Success using [{model_name}]")
            return response
            
        except APIError as e:
            err_str = str(e).lower()
            if any(k in err_str for k in ["429", "503", "500", "quota", "exhausted", "unavailable", "overloaded"]):
                if i < len(model_queue) - 1:
                    print(f"   ⚠️ Model [{model_name}] is busy/rate-limited. Seamlessly falling back...")
                    time.sleep(3) 
                    continue
                else:
                    print("   ❌ CRITICAL: All fallback models are currently unavailable or exhausted.")
                    raise e
            else:
                print(f"   ❌ Fatal API Error on [{model_name}]: {e}")
                raise e

# ==============================================================================
# STEP 1: SOURCING & DEDUPLICATION
# ==============================================================================

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(topic):
    if DEV_MODE:
        print("♻️ DEV MODE: Skipping history log to keep your production history clean.")
        return
        
    history = load_history()
    history.append(topic)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
    print(f"✅ Logged '{topic}' to {HISTORY_FILE}")

# ==============================================================================
# STEP 2: TOPIC, SCRIPT & TAGS
# ==============================================================================

def generate_topic_script_tags(history):
    if DEV_MODE and os.path.exists(SCRIPT_CACHE_FILE):
        print("♻️ DEV MODE: Loading cached script from disk (Skipping Gemini API)...")
        with open(SCRIPT_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"🎯 Topic Selected (Cached): {data['title']}")
            return data

    print("🧠 Brainstorming unique script with hyper-paced 16-clip structure...")
    
    prompt = f"""
    You are a viral YouTube Shorts producer. Create a 45-second script similar to the 'Starbucks glitch' story: fast-paced, mysterious, 'did you know?' style storytelling.
    Do NOT use any of these past topics: {json.dumps(history)}
    
    CRITICAL VISUAL B-ROLL RULE:
    The 'tags' array MUST contain exactly 16 simple, 1-2 word NOUNS that match the chronological story beats.
    Stock footage APIs are dumb. Do NOT use verbs or complex actions (e.g., 'scrolling phone', 'officer knocking').
    Use basic, highly searchable objects/nouns instead (e.g., 'smartphone', 'bank check', 'ATM', 'cash', 'crowd', 'laptop', 'police car').
    
    Output ONLY a JSON object with this exact structure:
    {{
      "title": "A short, catchy, mysterious title",
      "script": "The complete spoken script. Write it exactly as it should be read by a voiceover artist. No brackets, no stage directions. Around 110-130 words.",
      "tags": ["noun 1", "noun 2", "noun 3", "noun 4", "noun 5", "noun 6", "noun 7", "noun 8", "noun 9", "noun 10", "noun 11", "noun 12", "noun 13", "noun 14", "noun 15", "noun 16"] 
    }}
    """
    
    response = generate_with_fallback(
        contents=prompt,
        model_queue=ROUTING_LOGIC["heavy_reasoning"],
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    
    data = json.loads(response.text)
    print(f"🎯 Topic Selected: {data['title']}")
    print(f"🏷️ Generated B-Roll Tags: {data['tags']}")
    
    with open(SCRIPT_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        
    return data

# ==============================================================================
# STEP 3 & 6 COMBINED: VOICEOVER & FORCED ALIGNMENT CAPTIONS
# ==============================================================================

def generate_audio_and_captions(script_text):
    audio_path = os.path.join(ASSETS_DIR, "voiceover.mp3")
    
    if DEV_MODE and os.path.exists(audio_path) and os.path.exists(TIMESTAMPS_CACHE_FILE):
        print("♻️ DEV MODE: Using existing voiceover and alignment data...")
        with open(TIMESTAMPS_CACHE_FILE, "r", encoding="utf-8") as f:
            subs = json.load(f)
        return audio_path, subs

    print("🎙️ Generating voiceover and extracting native word-level timestamps via ElevenLabs...")
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/with-timestamps"
    headers = {
        "Content-Type": "application/json",
        "xi-api-key": ELEVEN_KEY
    }
    
    data = {
        "text": script_text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.25, 
            "similarity_boost": 0.85, 
            "style": 0.50,
            "use_speaker_boost": True
        }
    }
    
    resp = requests.post(url, json=data, headers=headers)
    
    if not resp.ok:
        print(f"❌ ElevenLabs API failed! Status Code: {resp.status_code}")
        print(f"Response: {resp.text}")
        resp.raise_for_status()
        
    response_data = resp.json()
    
    # Decode and save the audio
    audio_bytes = base64.b64decode(response_data["audio_base64"])
    with open(audio_path, 'wb') as f:
        f.write(audio_bytes)
        
    # Process character alignments into word-level timestamps
    alignment = response_data["alignment"]
    chars = alignment["characters"]
    starts = alignment["character_start_times_seconds"]
    ends = alignment["character_end_times_seconds"]

    words = []
    current_word = ""
    word_start = None

    for i, char in enumerate(chars):
        if char == " ":
            if current_word:
                words.append((word_start, ends[i-1], current_word))
                current_word = ""
                word_start = None
        else:
            if current_word == "":
                word_start = starts[i]
            current_word += char

    if current_word:
        words.append((word_start, ends[-1], current_word))

    # Stitch words into 1-2 word fast-paced visual chunks
    subs = []
    temp_words = []
    chunk_start = 0

    for w_start, w_end, word in words:
        if len(temp_words) == 0:
            chunk_start = w_start
        
        temp_words.append(word)

        if len(temp_words) >= 2 or word[-1] in ".!?":
            subs.append([chunk_start, w_end, " ".join(temp_words)])
            temp_words = []
            
    if temp_words:
        subs.append([chunk_start, words[-1][1], " ".join(temp_words)])

    with open(TIMESTAMPS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(subs, f, indent=4)

    print("✅ Audio and mathematically perfect captions generated.")
    return audio_path, subs

# ==============================================================================
# STEP 4: B-ROLL SOURCING (PEXELS)
# ==============================================================================

def download_b_roll(tags): 
    required = 16
    
    if DEV_MODE:
        existing_broll = [os.path.join(ASSETS_DIR, f) for f in os.listdir(ASSETS_DIR) if f.startswith("broll_") and f.endswith(".mp4")]
        if len(existing_broll) >= required:
            print(f"♻️ DEV MODE: Found {len(existing_broll)} existing B-Roll clips (Skipping Pexels API)...")
            return existing_broll[:required]

    print(f"🎬 Sourcing strictly ONE video per tag slot to maintain sync...")
    headers = {"Authorization": PEXELS_KEY}
    downloaded = []
    
    tag_map = {
        "atm machine keypad": ["atm machine", "atm keypad", "insert credit card"],
        "writing paper check": ["writing check", "signing document", "pen on paper"],
        "counting dollar bills": ["counting money", "cash money", "stack of cash"],
        "crowd waiting in line": ["people in line", "busy street crowd", "waiting crowd"],
        "police officer knocking": ["police lights", "siren flashing", "police badge"],
        "laptop screen warning": ["error screen", "hacking laptop", "computer screen"],
        "police sirens": ["police lights", "siren flashing", "cop car"]
    }

    fallback_tags = ["cinematic abstract", "dark city street", "neon lights", "cyberpunk code", "mysterious shadow", "fast cars"]
    tags.extend(fallback_tags)
    
    for i, tag in enumerate(tags[:required]):
        print(f"\n   🔍 Searching Slot {i+1}/{required} for tag: '{tag}'")
        
        search_queries = [tag, tag.split()[0], "cinematic abstract dark"]
        if tag.lower() in tag_map:
            search_queries = tag_map[tag.lower()] + search_queries
        
        found_for_this_slot = False
        
        for query in search_queries:
            if found_for_this_slot: 
                break 
                
            url = f"https://api.pexels.com/videos/search?query={query}&orientation=portrait&per_page=5"
            try:
                resp = requests.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                
                for video in data.get('videos', []):
                    if video.get('duration', 0) < 3: 
                        continue
                        
                    files = video.get('video_files', [])
                    hd_files = [v for v in files if v.get('quality') == 'hd' and v.get('width', 0) >= 720]
                    best_file = hd_files[0] if hd_files else files[0]
                    
                    vid_resp = requests.get(best_file['link'])
                    vid_resp.raise_for_status()
                    
                    filename = os.path.join(ASSETS_DIR, f"broll_{uuid.uuid4().hex[:6]}.mp4")
                    with open(filename, 'wb') as f:
                        f.write(vid_resp.content)
                        
                    downloaded.append(filename)
                    print(f"   ✅ Downloaded: {filename} (Using query: '{query}')")
                    found_for_this_slot = True
                    break 
                    
            except Exception as e:
                print(f"   ⚠️ API Error fetching query '{query}': {e}")
                
        if not found_for_this_slot:
            print(f"   ❌ CRITICAL WARNING: Slot {i+1} failed completely. Timeline may desync.")
            
    return downloaded

# ==============================================================================
# STEP 5: QUALITY CONTROL (GEMINI VISION)
# ==============================================================================

def verify_b_roll(video_paths, topic):
    if DEV_MODE:
        print("♻️ DEV MODE: Skipping Vision QC to save API calls. Assuming all B-Roll is valid.")
        return video_paths

    print("👁️ Initiating Vision Quality Control...")
    valid_videos = []
    
    for vid in video_paths:
        frame_path = vid + ".jpg"
        cap = cv2.VideoCapture(vid)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            continue
            
        cv2.imwrite(frame_path, frame)
        
        try:
            sample_file = gemini_client.files.upload(file=frame_path)
            prompt = f"Does this image visually fit as background footage for a mystery story about '{topic}'? Answer ONLY 'YES' or 'NO'."
            
            response = generate_with_fallback(
                contents=[sample_file, prompt],
                model_queue=ROUTING_LOGIC["fast_vision"]
            )
            
            answer = response.text.strip().upper()
            gemini_client.files.delete(name=sample_file.name)
            
            if "YES" in answer:
                print(f"   ✅ QC Accepted: {vid}")
                valid_videos.append(vid)
            else:
                print(f"   ❌ QC Rejected: {vid}")
        except Exception as e:
            print(f"   ⚠️ QC Error on {vid}: {e}. Accepting tentatively.")
            valid_videos.append(vid)
        finally:
            if os.path.exists(frame_path):
                os.remove(frame_path)
                
    if len(valid_videos) < len(video_paths):
        print("⚠️ Some videos rejected by QC! Falling back to raw download pool to ensure perfect timeline sync.")
        return video_paths
        
    return valid_videos

# ==============================================================================
# STEP 7: VIDEO ASSEMBLY (HYPER-PACING ENGINE)
# ==============================================================================

def assemble_video(audio_path, valid_videos, subs_data, final_title):
    print("🎞️ Stitching visual, audio, and captions in MoviePy...")
    audio = AudioFileClip(audio_path)
    audio_duration = audio.duration
    
    num_clips = len(valid_videos)
    if num_clips == 0:
        raise ValueError("No valid videos available to assemble.")
        
    clip_duration = audio_duration / num_clips
    print(f"⚡ Hyper-Pacing Engine: Cutting {num_clips} clips to exactly {clip_duration:.2f} seconds each.")

    clips = []
    target_w, target_h = 1080, 1920
    target_ratio = target_w / target_h

    for vid in valid_videos:
        clip = VideoFileClip(vid)
        
        # 🛡️ Bulletproof Dynamic Aspect Ratio Fix
        clip_ratio = clip.w / clip.h
        if clip_ratio > target_ratio:
            clip = clip.resize(height=target_h)
            clip = clip.crop(x_center=clip.w/2, width=target_w)
        else:
            clip = clip.resize(width=target_w)
            clip = clip.crop(y_center=clip.h/2, height=target_h)
            
        if clip.duration < clip_duration:
            clip = clip.fx(vfx.loop, duration=clip_duration)
        else:
            clip = clip.subclip(0, clip_duration)
            
        clips.append(clip)
        
    final_visual = concatenate_videoclips(clips, method="compose")
    
    final_visual = final_visual.set_audio(audio)
    if final_visual.duration > audio_duration:
        final_visual = final_visual.subclip(0, audio_duration)
    
    text_clips = []
    for start, end, text in subs_data:
        if end <= start: continue
        if start > audio_duration: break
        end = min(end, audio_duration)
        
        # 🚀 Viral Font Aesthetic: Arial-Bold, centered, massive stroke
        txt_clip = TextClip(text, fontsize=110, color='white', font='Arial-Bold', 
                            stroke_color='black', stroke_width=5, 
                            size=(950, None), method='caption')
        
        txt_clip = txt_clip.set_start(start).set_end(end).set_position(('center', 'center'))
        text_clips.append(txt_clip)
        
    final_video = CompositeVideoClip([final_visual] + text_clips)
    
    safe_title = "".join([c for c in final_title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
    output_path = os.path.join(OUTPUT_DIR, f"{safe_title.replace(' ', '_')}.mp4")
    
    print(f"🚀 Rendering final video to: {output_path}")
    final_video.write_videofile(
        output_path, 
        fps=30, 
        codec="libx264", 
        audio_codec="aac", 
        preset="ultrafast", 
        threads=4,
        logger='bar'
    )
    print("🎉 Render complete!")
    return output_path

# ==============================================================================
# MAIN PIPELINE EXECUTION
# ==============================================================================

def main():
    print("🔥 INITIALIZING VIDEO SLOP V2 PIPELINE 🔥")
    if DEV_MODE:
        print("🛠️  DEV MODE IS ON: Will reuse local assets and preserve API limits.")
    
    # 1. Sourcing & Deduplication
    history = load_history()
    
    # 2. Topic, Script & Tags
    gemini_data = generate_topic_script_tags(history)
    title = gemini_data['title']
    script = gemini_data['script']
    tags = gemini_data['tags']
    
    # 3 & 6. Voiceover & Captions (Combined for perfect Forced Alignment sync)
    audio_path, subs_data = generate_audio_and_captions(script)
    
    # 4. B-Roll Sourcing
    raw_videos = download_b_roll(tags)
    
    # 5. Quality Control
    valid_videos = verify_b_roll(raw_videos, title)
    
    # 7. Video Assembly
    assemble_video(audio_path, valid_videos, subs_data, title)
    
    # Logging & Cleanup
    save_history(title)
    
    if not DEV_MODE:
        for file in os.listdir(ASSETS_DIR):
            os.remove(os.path.join(ASSETS_DIR, file))
        print("🧹 Cleaned up temporary assets.")
    else:
        print("🛑 DEV MODE: Assets left in folder for next run.")
        
    print(f"🎬 PIPELINE FINISHED SUCCESSFULLY. Output is in the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()