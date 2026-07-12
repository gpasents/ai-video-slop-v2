import os
import json
import time
import uuid
import requests
import base64
import re
import argparse
import random
import math
import io
import hashlib
import shutil
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

# Using the brand new, officially supported Google GenAI SDK
from google import genai
from google.genai import types
from google.genai.errors import APIError

from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, ImageClip, CompositeVideoClip, concatenate_videoclips, CompositeAudioClip
import moviepy.video.fx.all as vfx
import moviepy.audio.fx.all as afx

# ==============================================================================
# CONFIGURATION & ENVIRONMENT SETUP
# ==============================================================================

# 🛑 DEVELOPMENT MODE TOGGLE 🛑
DEV_MODE = False

# ⚡ RENDER SPEED TOGGLE ("test" or "production") ⚡
RENDER_QUALITY = "production"

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
    "heavy_reasoning": ["gemini-3.1-flash-lite","gemini-3.5-flash","gemini-2.5-flash","gemini-2.5-flash-lite","gemini-3-flash"],
    "vision_tasks": ["gemini-3.1-flash-lite","gemini-3.5-flash","gemini-2.5-flash","gemini-2.5-flash-lite","gemini-3-flash"]
}

# Load keys as lists. Use commas in your .env file
GEMINI_KEYS = [k.strip() for k in os.environ.get("GEMINI_API_KEYS", os.environ.get("GEMINI_API_KEY", "")).split(",") if k.strip()]
ELEVEN_KEYS = [k.strip() for k in os.environ.get("ELEVENLABS_API_KEYS", os.environ.get("ELEVENLABS_API_KEY", "")).split(",") if k.strip()]
PEXELS_KEYS = [k.strip() for k in os.environ.get("PEXELS_API_KEYS", os.environ.get("PEXELS_API_KEY", "")).split(",") if k.strip()]
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "").strip()

if not GEMINI_KEYS or not ELEVEN_KEYS or not PEXELS_KEYS:
    print("❌ API Key Error: Missing one or more API key lists. Please check your .env file.")
    exit(1)

if not SERPER_API_KEY:
    print("❌ API Key Error: Missing SERPER_API_KEY. Please add it to your .env file.")
    exit(1)

current_gemini_idx = 0
current_eleven_idx = 0
current_pexels_idx = 0

# ==============================================================================
# OPTIMIZED WATERFALL ROUTER (WITH PROGRESSIVE BACKOFF & SMART BREAKS)
# ==============================================================================

def generate_with_fallback(contents, model_queue, config=None):
    global current_gemini_idx
    attempts = 0
    max_attempts = len(GEMINI_KEYS) * 3  
    backoff = 2
    
    while attempts < max_attempts:
        current_key = GEMINI_KEYS[current_gemini_idx]
        temp_client = genai.Client(api_key=current_key)
        
        for model_name in model_queue:
            try:
                response = temp_client.models.generate_content(
                    model=model_name, contents=contents, config=config
                )
                return response
            except APIError as e:
                err_str = str(e).lower()
                if any(k in err_str for k in ["429", "quota", "exhausted", "limit"]):
                    print(f"⚠️ Key Index [{current_gemini_idx}] rate-limited on {model_name}. Cycling to next key...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 15) 
                    break 
                
                print(f"⚠️ Model {model_name} failed on key index [{current_gemini_idx}]: {e}")
                time.sleep(1)
                continue
                
        current_gemini_idx = (current_gemini_idx + 1) % len(GEMINI_KEYS)
        attempts += 1
        
    raise Exception("❌ CRITICAL: Gemini API limits reached on all available keys and models after multiple retries.")

# ==============================================================================
# ROBUST JSON PARSER HELPER
# ==============================================================================

def clean_json_response(text):
    """Safely extracts and parses JSON even if the model wraps it in Markdown."""
    try:
        # First attempt a direct parse
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find a JSON block wrapped in markdown
    match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
            
    # As a last resort, try to extract everything between the first { and last }
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            return json.loads(text[start_idx:end_idx+1])
    except json.JSONDecodeError:
        pass
        
    print(f"⚠️ Failed to parse JSON from response. Raw text:\n{text}")
    raise ValueError("Could not parse JSON from the model's response.")

# ==============================================================================
# PROFILE MANAGEMENT
# ==============================================================================

def load_or_create_profile(profile_name):
    os.makedirs("profiles", exist_ok=True)
    profile_path = os.path.join("profiles", f"{profile_name}.json")
    
    if os.path.exists(profile_path):
        with open(profile_path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    print(f"⚠️ Profile '{profile_name}' not found. Auto-generating expanded Viral Variety profile...")
    
    default_profile = {
        "theme_name": "Viral Variety",
        "voice_id": "TX3LPaxmHKxFdv7VOQHJ", 
        "voice_model": "eleven_multilingual_v2",
        "voice_stability": 0.30,
        "voice_similarity": 0.55,
        "voice_style": 0.50,
        "voice_speed": 1.10,
        "bg_music_file": "bg_music.mp3",
        "bg_music_volume": 0.08,
        "visual_settings": {
            "font_family": "Arial-Black",
            "font_color": "#FFFF00",
            "stroke_color": "black",
            "caption_y_percentage": 0.60
        },
        "writer_prompt": """
You are a viral YouTube Shorts, TikTok, and Instagram Reels producer specializing in hyper-retention storytelling. 
Create a unique 45-second script concept based on fascinating facts, macabre history, bizarre science, or intriguing true stories.

Do NOT use any of these past topics: {history}

🚨 CRITICAL HOOK & ENDING RULE:
DO NOT start the script with "Imagine this", "Did you know", or by stating a year/date. 
The first 3 seconds MUST be a "Curiosity Loop"—an unanswered question or a bizarre contradiction that naturally forces the brain to seek closure. Do not just state a shocking fact; set up an immediate, unresolved mystery that the viewer HAS to watch to understand.
CRITICAL CTA ENDING: Instead of an abrupt cut, the script MUST conclude with a short, natural Call To Action (CTA). You must vary this CTA every time. Keep it punchy!

CRITICAL VISUAL B-ROLL RULE:
The 'tags' array MUST contain exactly 16 highly descriptive TWO-WORD search phrases (e.g., "arctic ice", "vintage shipwreck", "dark blizzard") that match the chronological story beats for Pexels. Avoid abstract terms.

You must generate exactly 5 natural attempts at the spoken script so our producer can pick the best one. Do not force specific differences or themes between them; simply try to write the strongest possible version based on the rules above, 5 times.

Output ONLY a JSON object with this exact structure:
{
  "title": "Internal working title",
  "topic_summary": "A 1-sentence factual description of the core historical event, person, or scientific phenomenon so we do not repeat it.",
  "script_variations": [
    "Attempt 1: The spoken script. Around 110-130 words.",
    "Attempt 2: Another natural attempt at the script.",
    "Attempt 3: Another natural attempt at the script.",
    "Attempt 4: Another natural attempt at the script.",
    "Attempt 5: Another natural attempt at the script."
  ],
  "tags": ["phrase1", "phrase2", "phrase3", "phrase4", "phrase5", "phrase6", "phrase7", "phrase8", "phrase9", "phrase10", "phrase11", "phrase12", "phrase13", "phrase14", "phrase15", "phrase16"],
  "metadata": {
    "youtube_shorts": {
        "title": "Engaging Short Title",
        "description": "Fascinating description of the video. #tags",
        "tags": ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7"],
        "made_for_kids": false,
        "category_id": "24"
    },
    "tiktok": {
        "caption": "Engaging TikTok caption here #fyp #tags",
        "disable_comment": false,
        "disable_duet": false,
        "disable_stitch": false,
        "video_cover_timestamp_ms": 1500,
        "brand_content_toggle": false
    },
    "instagram_reels": {
        "caption": "Engaging IG Reel caption here. #reelsviral #tags",
        "share_to_feed": true,
        "cover_image_timestamp_ms": 1500
    }
  }
}
""",
        "script_qa_prompt": """
You are an expert Executive Producer for YouTube Shorts.
Review the following script variations and select the one with the strongest curiosity hook, best pacing, and highest retention probability.

Script Variations:
{variations}

Output ONLY a JSON object with the index (0-based) of the best script and a 1-sentence reason.
{
  "best_index": 2,
  "reason": "Strongest opening hook that introduces an immediate, unresolved paradox."
}
""",
        "audio_director_prompt": """
You are an Audio Director preparing scripts for ElevenLabs AI narration. Your only job is to adjust HOW it reads aloud so a text-to-speech engine produces natural, confident, human-sounding narration.

You will receive a JSON object from a scriptwriter. Take the 'script' field and rewrite it for voice performance.

🚨 CRITICAL PAUSE RULE:
Do NOT use ellipses ("..."). Use commas for quick breaths and periods for full stops. Vary sentence lengths for a natural human cadence.

🚨 CRITICAL DYNAMIC CAPTION RULE:
You must explicitly target exactly 4 to 7 crucial, high-impact shock words to be completely UPPERCASE. These words will trigger custom color changes in production (e.g., 'That paint was PURE radium'). Do not over-capitalize.

🚨 CRITICAL PRONUNCIATION RULE:
Spell out ambiguous numbers, dates, and abbreviations exactly how they should be spoken (e.g., '1931' → 'nineteen thirty-one').

Raw Script:
{script}

Output ONLY a JSON object with this exact structure:
{
  "directed_script": "The continuous, fast-paced script here."
}
""",
        "editor_prompt": """
You are a master Cinematic Video Editor and Sound Designer.
I am going to provide you with the exact, word-by-word timestamps of a voiceover.
Your job is to strategically pick 5 to 9 key moments in the timeline to flash a highly specific historical image on screen, AND assign an accompanying cinematic sound effect (SFX) to amplify the impact.

Rules:
1. Provide the exact 'start' time in seconds for the visual asset.
2. Provide a 'search_query' for Google Images. It MUST be historically concrete and highly specific (e.g., "SS Baychimo stuck in ice 1931 photo", "Victor Lustig police mugshot"). NEVER use abstract concepts.
3. Provide a 'duration' in seconds for the image overlay (usually 2.0 to 3.5).
4. Provide an 'image_type' ("person" or "object").
5. Provide an 'sfx_trigger' classification based on the emotional beat. You MUST choose ONLY from these exact words: "whoosh", "thud", "newspaper", "pop", "shutter", or "swoosh".

Voiceover Timestamps:
{timestamps}

Output ONLY a JSON object with this exact structure:
{
  "effects": [
    {
      "start": 1.45,
      "search_query": "SS Baychimo ghost ship arctic historical photo",
      "duration": 2.5,
      "image_type": "object",
      "sfx_trigger": "thud"
    }
  ]
}
""",
        "vision_selection_prompt": """
You are a Cinematic Art Director.
I will provide you with multiple candidate images fetched for the historical search query: '{query}'.
Your task is to select the single best image that is the highest quality, most visually striking, most historically relevant, and most cinematic. Avoid heavy watermarks, abstract illustrations, or completely irrelevant icons.
Return ONLY a JSON object with the 0-based index of the chosen image.
{
  "best_index": 0
}
""",
        "vision_qa_prompt": """
You are an expert Quality Assurance director for a documentary video.
I am providing you with a sequence of images and their intended search queries.

Your job is to flag any image that fails EITHER of these two rules:
1. Duplicate: The image is visually identical to an earlier image in the sequence.
2. Irrelevant or Low Quality: The image clearly does NOT match its intended search query, is an abstract drawing/icon instead of a photo, contains heavy watermarks, or is completely nonsensical.

Return ONLY a JSON object containing a list of the 0-based indices of the images that fail these rules and must be replaced.
Strict format: {"bad_indices": [1, 3]}
"""
    }
    
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(default_profile, f, indent=4)
        
    return default_profile

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

def evaluate_best_script(script_variations, profile):
    print(f"🧠 [STAGE 1.2] Executive Producer QA evaluating {len(script_variations)} script variations...")
    prompt = profile["script_qa_prompt"].replace("{variations}", json.dumps(script_variations, indent=2))
    
    try:
        response = generate_with_fallback(
            contents=prompt,
            model_queue=ROUTING_LOGIC["heavy_reasoning"],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2
            )
        )
        
        data = clean_json_response(response.text)
        best_idx = data.get("best_index", 0)
        
        if not isinstance(best_idx, int) or best_idx < 0 or best_idx >= len(script_variations):
            best_idx = 0
            
        print(f"  🏆 Selected Variation {best_idx + 1} - Reason: {data.get('reason', 'N/A')}")
        return script_variations[best_idx]
    except Exception as e:
        print(f"  ⚠️ Script QA failed: {e}. Defaulting to first variation.")
        return script_variations[0]

def generate_topic_script_tags(profile, history, script_cache_file, is_batching):
    if DEV_MODE and not is_batching and os.path.exists(script_cache_file):
        print("♻️ DEV MODE: Loading script from cache...")
        with open(script_cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    print("🧠 [STAGE 1] Brainstorming unique script variations and generating viral properties...")
    prompt = profile["writer_prompt"].replace("{history}", json.dumps(history))
    
    response = generate_with_fallback(
        contents=prompt,
        model_queue=ROUTING_LOGIC["heavy_reasoning"],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.8 
        )
    )
    
    data = clean_json_response(response.text)
    print(f"🎯 Topic Selected: {data['title']}")
    
    # Run Script QA Selection
    variations = data.get('script_variations', [])
    if not variations and 'script' in data:
        variations = [data['script']]
    elif not variations:
        variations = ["Emergency fallback script."]
        
    best_script = evaluate_best_script(variations, profile)
    data['script'] = best_script
    
    with open(script_cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        
    return data

def direct_audio_script(raw_script, profile, audio_dir_cache_file, is_batching):
    if DEV_MODE and not is_batching and os.path.exists(audio_dir_cache_file):
        print("♻️ DEV MODE: Loading directed audio script from cache...")
        with open(audio_dir_cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    print("🎧 [STAGE 1.5] Audio Director is injecting pacing and emotion cues for ElevenLabs...")
    prompt = profile["audio_director_prompt"].replace("{script}", raw_script)
    
    response = generate_with_fallback(
        contents=prompt,
        model_queue=ROUTING_LOGIC["heavy_reasoning"],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.4
        )
    )
    
    data = clean_json_response(response.text)
    
    with open(audio_dir_cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        
    return data

def generate_audio_and_captions(script_text, profile, audio_path, timestamps_cache_file, is_batching):
    global current_eleven_idx
    
    if DEV_MODE and not is_batching and os.path.exists(audio_path) and os.path.exists(timestamps_cache_file):
        print("♻️ DEV MODE: Loading audio and timestamps from cache...")
        with open(timestamps_cache_file, "r", encoding="utf-8") as f:
            subs = json.load(f)
        return audio_path, subs

    print("🎙️ [STAGE 2] Generating ElevenLabs voiceover and extracting timeline...")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{profile['voice_id']}/with-timestamps"
    data = {
        "text": script_text,
        "model_id": profile.get("voice_model", "eleven_multilingual_v2"),
        "voice_settings": {
            "stability": profile.get("voice_stability", 0.30), 
            "similarity_boost": profile.get("voice_similarity", 0.55), 
            "style": profile.get("voice_style", 0.80),
            "speed": profile.get("voice_speed", 1.15),
            "use_speaker_boost": True
        }
    }
    
    debug_path = os.path.join(os.path.dirname(audio_path), "elevenlabs_debug_request.json")
    try:
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump({"url": url, "payload": data}, f, indent=4)
    except Exception as e:
        print(f"   ⚠️ Could not save ElevenLabs debug file: {e}")
    
    start_idx = current_eleven_idx
    while True:
        headers = {"Content-Type": "application/json", "xi-api-key": ELEVEN_KEYS[current_eleven_idx]}
        resp = requests.post(url, json=data, headers=headers)
        
        if resp.status_code in [401, 402, 429]:
            current_eleven_idx = (current_eleven_idx + 1) % len(ELEVEN_KEYS)
            if current_eleven_idx == start_idx:
                raise Exception("❌ CRITICAL: All ElevenLabs keys are exhausted.")
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
            if not char.isalnum() and char not in ["'", "’"]:
                if current_word:
                    end_idx = i - 1 if i > 0 else 0
                    words.append((word_start, ends[end_idx], current_word))
                    current_word, word_start = "", None
            else:
                if current_word == "": word_start = starts[i]
                current_word += char
                
        if current_word: 
            words.append((word_start, ends[-1], current_word))

        subs = []
        for w_start, w_end, word in words:
            if word: 
                subs.append([w_start, w_end, word])

        with open(timestamps_cache_file, "w", encoding="utf-8") as f:
            json.dump(subs, f, indent=4)
            
        return audio_path, subs

def generate_editor_effects(subs_data, profile, effects_cache_file, is_batching):
    if DEV_MODE and not is_batching and os.path.exists(effects_cache_file):
        print("♻️ DEV MODE: Loading editor effects from cache...")
        with open(effects_cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    print("🎬 [STAGE 3] Sending precise timeline to AI Director to plan visual impacts...")
    
    compact_timeline = [{"start": w[0], "word": w[2]} for w in subs_data]
    timeline_str = json.dumps(compact_timeline)
    
    prompt = profile["editor_prompt"].replace("{timestamps}", timeline_str)
    
    response = generate_with_fallback(
        contents=prompt,
        model_queue=ROUTING_LOGIC["heavy_reasoning"],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.4
        )
    )
    
    data = clean_json_response(response.text)
    effects = data.get("effects", [])
    
    for effect in effects:
        print(f"  ⚡ AI Editor queued image '{effect.get('search_query')}' (Type: {effect.get('image_type', 'object')}) at {effect.get('start')}s")
    
    with open(effects_cache_file, "w", encoding="utf-8") as f:
        json.dump(effects, f, indent=4)
        
    return effects

# --- STAGE 4: SERPER API MULTI-IMAGE FETCHER ---
def fetch_serper_image_pool(search_query, output_dir, pool_size=3, ignored_urls=None):
    if ignored_urls is None:
        ignored_urls = []
        
    print(f"🔍 Searching Google Images for candidate pool: '{search_query}'")
    url = "https://google.serper.dev/images"
    payload = json.dumps({"q": search_query})
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status() 
        data = response.json()
        
        images = data.get("images", [])
        if not images:
            print(f"  ⚠️ No images found on Google for '{search_query}'.")
            return [], []
            
        safe_name = "".join([c for c in search_query if c.isalnum()]).strip()
        
        downloaded_paths = []
        downloaded_urls = []

        for img_data_entry in images:
            if len(downloaded_paths) >= pool_size:
                break
                
            image_url = img_data_entry.get("imageUrl")
            if not image_url or image_url in ignored_urls:
                continue
                
            url_hash = hashlib.md5(image_url.encode('utf-8')).hexdigest()[:8]
            filename = f"serper_{safe_name}_{url_hash}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            if os.path.exists(filepath):
                 downloaded_paths.append(filepath)
                 downloaded_urls.append(image_url)
                 continue

            try:
                browser_headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                
                img_resp = requests.get(image_url, headers=browser_headers, timeout=10)
                img_resp.raise_for_status() 
                
                img_bytes = io.BytesIO(img_resp.content)
                img = Image.open(img_bytes)
                
                if img.mode in ("RGBA", "P") or img.mode != "RGB":
                    img = img.convert("RGB")
                    
                img.save(filepath, "JPEG")
                downloaded_paths.append(filepath)
                downloaded_urls.append(image_url)
                
            except Exception:
                continue
        
        return downloaded_paths, downloaded_urls
        
    except Exception as e:
        print(f"  ⚠️ Serper API request failed for '{search_query}': {e}")
        return [], []

def select_best_candidate_image(image_paths, search_query, profile):
    if not image_paths:
        return None, 0
    if len(image_paths) == 1:
        return image_paths[0], 0

    print(f"🧠 [STAGE 4.1] Art Director evaluating {len(image_paths)} image candidates for '{search_query}'...")
    prompt = profile.get("vision_selection_prompt", "...").replace("{query}", search_query)
    contents = [prompt]

    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path)
            img.thumbnail((512, 512))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=75)
            contents.append(f"Candidate {i}:")
            contents.append(types.Part.from_bytes(data=img_byte_arr.getvalue(), mime_type="image/jpeg"))
        except Exception as img_err:
            print(f"  ⚠️ Thumbnail processing failed for candidate {i}: {img_err}")
            pass
            
    try:
        response = generate_with_fallback(
            contents=contents,
            model_queue=ROUTING_LOGIC["vision_tasks"],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        data = clean_json_response(response.text)
        best_idx = data.get("best_index", 0)
        
        if not isinstance(best_idx, int) or best_idx < 0 or best_idx >= len(image_paths):
            best_idx = 0
            
        print(f"  🏆 Selected Candidate {best_idx} for '{search_query}'")
        return image_paths[best_idx], best_idx
        
    except Exception as e:
        print(f"  ⚠️ Art Director QA skipped (API issue): {e}. Defaulting to first image.")
        return image_paths[0], 0

# ==============================================================================
# UPGRADED STAGE 4.5: GEMINI VISION QUALITY ASSURANCE (QA)
# ==============================================================================
def analyze_and_filter_images(matched_effects, assets_dir, profile):
    valid_effects = [e for e in matched_effects if e.get('local_path') and os.path.exists(e['local_path'])]
    if len(valid_effects) == 0:
        return matched_effects

    print("🧠 [STAGE 4.5] Global QA Check (Deduplication & Flow Relevance)...")
    
    prompt = profile.get("vision_qa_prompt", "You are a QA Director. Flag indices of bad or identical images as JSON {'bad_indices': []}.")
    contents = [prompt]
    
    for i, effect in enumerate(valid_effects):
        try:
            img = Image.open(effect['local_path'])
            img.thumbnail((512, 512)) 
            
            if img.mode != "RGB":
                img = img.convert("RGB")
                
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=75)
            img_bytes = img_byte_arr.getvalue()
            
            contents.append(f"Image {i} (Intended Query: '{effect.get('search_query', 'Unknown')}'):")
            contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
        except Exception as img_err:
            pass
        
    try:
        response = generate_with_fallback(
            contents=contents,
            model_queue=ROUTING_LOGIC["vision_tasks"],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", 
                temperature=0.1
            )
        )
        data = clean_json_response(response.text)
        bad_indices = data.get("bad_indices", [])
        
        if not bad_indices:
            print("  ✅ QA confirms sequence is unique and ready!")
            return matched_effects
            
        print(f"  ⚠️ QA flagged bad/duplicate images at sequence indices: {bad_indices}. Fetching replacements...")
        
        for idx in bad_indices:
            if isinstance(idx, int) and 0 <= idx < len(valid_effects):
                bad_effect = valid_effects[idx]
                
                # Fetch a new pool, making sure we ignore the ones we already tried
                paths, urls = fetch_serper_image_pool(
                    bad_effect['search_query'], 
                    assets_dir,
                    pool_size=3,
                    ignored_urls=bad_effect.get('ignored_urls', [])
                )
                
                if paths:
                    new_path, new_idx = select_best_candidate_image(paths, bad_effect['search_query'], profile)
                    
                    for effect in matched_effects:
                        if effect == bad_effect:
                            effect['local_path'] = new_path
                            new_url = urls[new_idx] if urls else None
                            effect['source_url'] = new_url
                            if new_url: effect['ignored_urls'].append(new_url)
                            break
                            
        return matched_effects

    except Exception as e:
        print(f"  ⚠️ AI QA skipped (Transient API/Quota issue): {e}. Proceeding safe.")
        return matched_effects

def get_pexels_data(url):
    global current_pexels_idx
    start_idx = current_pexels_idx
    while True:
        headers = {"Authorization": PEXELS_KEYS[current_pexels_idx]}
        resp = requests.get(url, headers=headers)
        if resp.status_code == 429:
            current_pexels_idx = (current_pexels_idx + 1) % len(PEXELS_KEYS)
            if current_pexels_idx == start_idx:
                raise Exception("❌ Pexels API exhausted on all keys.")
            time.sleep(2)
            continue
        resp.raise_for_status()
        return resp.json()

def download_b_roll(tags, assets_dir, is_batching): 
    required = 16
    existing = [os.path.join(assets_dir, f) for f in os.listdir(assets_dir) if f.startswith("broll_") and f.endswith(".mp4")]
    
    if DEV_MODE and not is_batching and len(existing) >= required:
        return sorted(existing)[:required]

    print(f"🎬 Sourcing strictly ONE video per tag slot...")
    downloaded = []
    
    for i, tag in enumerate(tags[:required]):
        primary_keyword = tag.split()[0] if " " in tag else tag 

        search_queries = [
            tag, 
            primary_keyword, 
            "mystery",       
            "abstract dark"  
        ]
        
        safe_tag = "".join([c for c in tag if c.isalnum() or c == ' ']).strip().replace(' ', '_')
        if not safe_tag: safe_tag = "fallback"
        
        slot_filled = False
        for query in search_queries:
            if slot_filled: break 
            
            url = f"https://api.pexels.com/videos/search?query={query}&orientation=portrait&per_page=15"
            try:
                data = get_pexels_data(url)
                valid_videos_in_query = [v for v in data.get('videos', []) if v.get('duration', 0) >= 3]
                
                if valid_videos_in_query:
                    video = random.choice(valid_videos_in_query)
                    
                    files = video.get('video_files', [])
                    hd_files = [v for v in files if v.get('quality') == 'hd' and v.get('width', 0) >= 720]
                    best_file = hd_files[0] if hd_files else files[0]
                    
                    vid_resp = requests.get(best_file['link'])
                    final_filename = os.path.join(assets_dir, f"broll_{i:02d}_{safe_tag}_{uuid.uuid4().hex[:4]}.mp4")
                    
                    with open(final_filename, 'wb') as f: 
                        f.write(vid_resp.content)
                    
                    downloaded.append(final_filename)
                    print(f"   ✅ Slot {i+1}/16 filled: '{query}'")
                    slot_filled = True
                    break
                        
            except Exception:
                pass
                
        if not slot_filled:
            print(f"   ⚠️ Exhausted options for '{tag}'. Using emergency fallback.")
            if downloaded:
                fallback_name = os.path.join(assets_dir, f"broll_{i:02d}_emergency_fallback_{uuid.uuid4().hex[:4]}.mp4")
                shutil.copy(downloaded[-1], fallback_name)
                downloaded.append(fallback_name)

    return downloaded

def assemble_video(audio_path, bg_music_path, valid_videos, subs_data, matched_effects, final_title, output_dir, profile):
    print("🎞️ Stitching visual, audio, captions, and EFFECTS in MoviePy...")
    
    # ---------------------------
    # SETUP MASTER AUDIO TRACKS
    # ---------------------------
    audio_tracks = []
    audio = AudioFileClip(audio_path)
    audio_duration = audio.duration
    audio_tracks.append(audio)

    bg_vol = profile.get("bg_music_volume", 0.08)
    bg_clip = None
    if bg_music_path and os.path.exists(bg_music_path):
        bg_clip = AudioFileClip(bg_music_path).fx(afx.volumex, bg_vol)
        bg_clip = afx.audio_loop(bg_clip, duration=audio_duration)
        audio_tracks.append(bg_clip)

    if RENDER_QUALITY == "test":
        target_w, target_h, render_fps, font_size, stroke_thickness, offset_shadow = 540, 960, 30, 50, 7, 4
    else:
        target_w, target_h, render_fps, font_size, stroke_thickness, offset_shadow = 1080, 1920, 60, 98, 15, 8

    vis_settings = profile.get("visual_settings", {})
    font_choice = vis_settings.get("font_family", "Arial-Black")
    font_color = vis_settings.get("font_color", "#FFFF00")
    stroke_col = vis_settings.get("stroke_color", "black")
    y_perc = vis_settings.get("caption_y_percentage", 0.60)

    clip_duration = audio_duration / len(valid_videos) if valid_videos else audio_duration
    
    clips = []
    base_clips = []
    
    for vid in valid_videos:
        base_clip = VideoFileClip(vid)
        base_clips.append(base_clip)
        
        clip = base_clip.without_audio().set_fps(render_fps)
        w, h = clip.size
        
        scale_w = target_w / float(w)
        scale_h = target_h / float(h)
        scale_factor = max(scale_w, scale_h) 
        
        clip = clip.resize(scale_factor)
        clip = clip.crop(x_center=clip.w/2.0, y_center=clip.h/2.0, width=target_w, height=target_h)
            
        if clip.duration < clip_duration:
            clip = clip.fx(vfx.loop, duration=clip_duration)
        else:
            start_time = 1.0 if clip.duration >= (clip_duration + 1.0) else 0.0
            clip = clip.subclip(start_time, start_time + clip_duration)
            
        clips.append(clip)
        
    final_visual = concatenate_videoclips(clips, method="compose")
    
    # --- CAPTIONS ---
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
        
        current_font_color = font_color
        
        txt_shadow = TextClip(raw_text, fontsize=font_size, color=stroke_col, font=font_choice, stroke_color=stroke_col, stroke_width=stroke_thickness, kerning=tight_kerning, method='label', align='center')
        txt_stroke = TextClip(raw_text, fontsize=font_size, color=stroke_col, font=font_choice, stroke_color=stroke_col, stroke_width=stroke_thickness, kerning=tight_kerning, method='label', align='center')
        txt_fill = TextClip(raw_text, fontsize=font_size, color=current_font_color, font=font_choice, stroke_width=0, kerning=tight_kerning, method='label', align='center')
        
        box_w, box_h = max(txt_shadow.w, txt_stroke.w, txt_fill.w) + 40, max(txt_shadow.h, txt_stroke.h, txt_fill.h) + 40
        cx, cy = (box_w - txt_stroke.w) / 2, (box_h - txt_stroke.h) / 2
        
        txt_shadow = txt_shadow.set_position((cx + offset_shadow, cy + offset_shadow))
        txt_stroke = txt_stroke.set_position((cx, cy))
        txt_fill = txt_fill.set_position(((box_w - txt_fill.w) / 2, (box_h - txt_fill.h) / 2))
        
        word_clip = CompositeVideoClip([txt_shadow, txt_stroke, txt_fill], size=(box_w, box_h)).resize(snappy_pop).set_position(('center', caption_y_pos)).set_start(start).set_end(end)
        text_clips.append(word_clip)

    # --- EFFECTS LAYERING (PRESERVING ASPECT RATIO) ---
    effect_clips = []
    sfx_clips = []
    
    sfx_dir = "sfx"
    
    sfx_mapping = {
        "whoosh": "dragon-studio-simple-whoosh-382724.mp3",
        "thud": "dragon-studio-thud-sound-effect-405470.mp3",
        "newspaper": "floraphonic-newspaper-foley-4-196721.mp3",
        "pop": "pop.mp3",
        "shutter": "shutter.mp3",
        "swoosh": "swoosh.mp3"
    }
    
    def image_overshoot_pop(t):
        if t < 0.07: return 0.10 + (1.10 / 0.07) * t
        elif t < 0.14: return 1.20 - (0.25 / 0.07) * (t - 0.07)
        elif t < 0.20: return 0.95 + (0.05 / 0.06) * (t - 0.14)
        return 1.0

    for effect in matched_effects:
        start_time = float(effect.get('start', 0.0))
        image_type = effect.get('image_type', 'object').lower()
        
        sfx_type = effect.get('sfx_trigger', 'whoosh').lower()
        actual_filename = sfx_mapping.get(sfx_type, "dragon-studio-simple-whoosh-382724.mp3") 
        custom_sfx_path = os.path.join(sfx_dir, actual_filename)
        shutter_sfx_path = os.path.join(sfx_dir, "shutter.mp3")
        
        if os.path.exists(custom_sfx_path):
            try:
                sfx_clip = AudioFileClip(custom_sfx_path).set_start(start_time).fx(afx.volumex, 0.1)
                sfx_clips.append(sfx_clip)
            except Exception as e:
                pass
        elif image_type == 'person' and os.path.exists(shutter_sfx_path):
            try:
                sfx_clip = AudioFileClip(shutter_sfx_path).set_start(start_time).fx(afx.volumex, 0.1)
                sfx_clips.append(sfx_clip)
            except Exception as e:
                pass
                
        if effect.get('local_path') and os.path.exists(effect['local_path']):
            try:
                fx_clip = ImageClip(effect['local_path'])
                
                max_w = int(target_w * 0.8)
                max_h = int(target_h * 0.45)
                
                w, h = fx_clip.size
                scale_factor = min(max_w / float(w), max_h / float(h))
                
                fx_clip = fx_clip.resize(scale_factor)
                
                if fx_clip.mask is None:
                    fx_clip = fx_clip.add_mask()
                    
                base_w, base_h = fx_clip.size 
                duration = float(effect.get('duration', 2.0))
                
                def get_dynamic_pos(t, bh=base_h):
                    scale = image_overshoot_pop(t)
                    curr_h = bh * scale
                    y = (target_h * 0.35) - (curr_h / 2)
                    return ('center', y)
                
                fx_clip = (fx_clip
                           .set_start(start_time)
                           .set_duration(duration)
                           .resize(image_overshoot_pop) 
                           .set_position(get_dynamic_pos) 
                           .crossfadein(0.15) 
                           .crossfadeout(0.2)) 
                
                effect_clips.append(fx_clip)
            except Exception as e:
                print(f"  ⚠️ Skipping image {effect.get('local_path')} due to render error: {e}")
        
    if sfx_clips:
        audio_tracks.extend(sfx_clips)
        
    final_mixed_audio = CompositeAudioClip(audio_tracks)
    final_visual = final_visual.set_audio(final_mixed_audio).subclip(0, audio_duration)
    
    final_video = CompositeVideoClip([final_visual] + effect_clips + text_clips)
    
    safe_title = "".join([c for c in final_title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
    base_filename = safe_title.replace(' ', '_')
    output_path = os.path.join(output_dir, f"{base_filename}.mp4")
    
    final_video.write_videofile(
        output_path, 
        fps=render_fps, 
        codec="libx264", 
        audio_codec="aac", 
        preset="fast", 
        ffmpeg_params=["-crf", "23"], 
        threads=4, 
        logger='bar'
    )
    
    final_video.close()
    final_visual.close()
    audio.close()
    if bg_clip: bg_clip.close()
    for clip in clips: clip.close()
    for base in base_clips: base.close() 
    for sfx in sfx_clips: sfx.close()

    print("🎉 Render complete and resources released.")
    return output_path, base_filename

# ==============================================================================
# SURGICAL CLEANUP
# ==============================================================================

def cleanup_workspace(assets_dir, bg_music_filename):
    print("🧹 Running surgical cleanup (Leaving history, bg_music, and debug files intact)...")
    
    time.sleep(1.5) 
    
    for file in os.listdir(assets_dir):
        if file == "elevenlabs_debug_request.json":
            continue
            
        if file.endswith(".mp4") or file.endswith(".jpg") or file.endswith(".png") or file == "voiceover.mp3" or "cache" in file:
            if file != bg_music_filename:
                try: 
                    os.remove(os.path.join(assets_dir, file))
                except Exception as e: 
                    print(f"  ⚠️ Could not delete {file}: {e}")

# ==============================================================================
# MAIN BATCH PIPELINE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Batch Video Generator")
    parser.add_argument("--profile", type=str, default="viral_variety", help="Name of the channel profile to load")
    parser.add_argument("--count", type=int, default=1, help="Number of videos to generate in this batch")
    args = parser.parse_args()

    print(f"🔥 INITIALIZING BATCH PIPELINE: Profile [{args.profile}] | Target Count: {args.count} 🔥")

    profile = load_or_create_profile(args.profile)
    
    assets_dir = f"assets_{args.profile}"
    base_output_dir = f"output_{args.profile}"
    history_file = os.path.join(assets_dir, "history.json")
    script_cache = os.path.join(assets_dir, "script_cache.json")
    audio_dir_cache = os.path.join(assets_dir, "audio_director_cache.json") 
    timestamps_cache = os.path.join(assets_dir, "timestamps_cache.json")
    effects_cache = os.path.join(assets_dir, "effects_cache.json")
    local_bg_music = os.path.join(assets_dir, profile["bg_music_file"])
    
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(base_output_dir, exist_ok=True)

    for i in range(args.count):
        print(f"\n=======================================================")
        print(f"🎬 STARTING VIDEO {i+1} OF {args.count}")
        print(f"=======================================================")
        
        is_batching = args.count > 1 
        history = load_history(history_file)
        
        # STAGE 1 & 1.2: Writer & Script QA Evaluation
        gemini_data = generate_topic_script_tags(profile, history, script_cache, is_batching)
        title = gemini_data['title']
        topic_summary = gemini_data.get('topic_summary', 'No summary provided')
        history_entry = f"Title: '{title}' - Topic: {topic_summary}"
        
        safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
        base_filename = safe_title.replace(' ', '_')
        video_out_dir = os.path.join(base_output_dir, base_filename)
        os.makedirs(video_out_dir, exist_ok=True)
        
        # STAGE 1.5: Audio Director 
        directed_script_data = direct_audio_script(gemini_data['script'], profile, audio_dir_cache, is_batching)
        final_spoken_script = directed_script_data.get('directed_script', gemini_data['script'])

        # STAGE 2: Audio & Timeline Extraction
        audio_path, subs_data = generate_audio_and_captions(
            final_spoken_script, profile, os.path.join(assets_dir, "voiceover.mp3"), timestamps_cache, is_batching
        )
        
        # STAGE 3: Director / Video Editor AI
        matched_effects = generate_editor_effects(subs_data, profile, effects_cache, is_batching)
        
        # STAGE 4 & 4.1: Initial Resource Sourcing & Candidate Image Selection QA
        for effect in matched_effects:
            if effect.get('search_query'):
                paths, urls = fetch_serper_image_pool(effect['search_query'], assets_dir, pool_size=3)
                if paths:
                    best_path, best_idx = select_best_candidate_image(paths, effect['search_query'], profile)
                    effect['local_path'] = best_path
                    effect['source_url'] = urls[best_idx] if urls else None
                    effect['ignored_urls'] = urls
        
        # STAGE 4.5: Upgraded Gemini QA Sequence Deduplication
        matched_effects = analyze_and_filter_images(matched_effects, assets_dir, profile)
        
        bg_music_path = local_bg_music if os.path.exists(local_bg_music) else None
        valid_videos = download_b_roll(gemini_data['tags'], assets_dir, is_batching)
        
        # STAGE 5: Rendering
        _, returned_base_filename = assemble_video(audio_path, bg_music_path, valid_videos, subs_data, matched_effects, title, video_out_dir, profile)
        
        if 'metadata' in gemini_data:
            metadata_output_path = os.path.join(video_out_dir, f"{returned_base_filename}.json")
            with open(metadata_output_path, "w", encoding="utf-8") as meta_f:
                json.dump(gemini_data['metadata'], meta_f, indent=4)
            print(f"📦 Paired viral properties saved to: {metadata_output_path}")
        
        save_history(history_file, history_entry)
        
        if i < args.count - 1 or not DEV_MODE:
            cleanup_workspace(assets_dir, profile["bg_music_file"])

    print(f"\n🎉 BATCH COMPLETE! Generated {args.count} videos and metadata packets in '{base_output_dir}'.")

if __name__ == "__main__":
    main()