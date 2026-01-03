import os
import sys
import time
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, RequestBlocked, VideoUnavailable, VideoUnplayable, AgeRestricted
from deep_translator import GoogleTranslator
from urllib.parse import urlparse, parse_qs

def get_video_id(youtube_url):
    """
    Extract the video ID from a YouTube URL.
    """
    parsed_url = urlparse(youtube_url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]  
    elif parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        query_params = parse_qs(parsed_url.query)
        return query_params.get('v', [None])[0]  
    return None

def get_any_transcript(video_id, limit_duration=1795):
    """
    Fetch transcript for a video. 
    Returns transcript list on success, None if no transcript available.
    Raises RequestBlocked if API rate limit is hit.
    """
    ytt_api = YouTubeTranscriptApi()

    # Retrieve the available transcripts
    transcript_list = ytt_api.list(video_id)
    transcript = None

    # Iterate over all available transcripts
    for transcript_obj in transcript_list:
        if not transcript_obj.is_generated:
            transcript = transcript_obj.fetch()
            break
    if transcript is None:
        for transcript_obj in transcript_list:
            if transcript_obj.is_generated:
                transcript = transcript_obj.fetch()
                print(f"Using auto-generated transcript in {transcript_obj.language}")
                break

    if transcript is None:
        print(f"No transcript available for video ID {video_id}")
        return None
    
    limited_transcript = [entry for entry in transcript if entry.start <= limit_duration]
    return limited_transcript

def translate_transcript(transcript, dest_lang='en'):
    """
    Translate the transcript into a specified language.
    """
    translator = GoogleTranslator(target=dest_lang)
    translated_transcript = []
    for entry in transcript:
        try:
            if entry.text is not None:
                translated_text = translator.translate(entry.text)
                translated_transcript.append({'text': translated_text, 'start': entry.start, 'duration': entry.duration})
        except Exception as e:
            print(f"An error occurred during translation: {e}")
    return translated_transcript

def transcript_to_text(transcript):
    """
    Convert transcript entries to a single text string.
    """
    lines = [entry['text'] for entry in transcript if entry['text'] is not None]
    return ' '.join(lines)


def save_transcript_to_tsv(video_id, transcript_text, output_file):
    """
    Append a single transcript to the TSV file.
    Creates the file with headers if it doesn't exist.
    """
    file_exists = os.path.exists(output_file)
    df = pd.DataFrame([(video_id, transcript_text)], columns=['video_id', 'transcript'])
    df.to_csv(output_file, sep='\t', index=False, encoding='utf-8', mode='a', header=not file_exists)
    print(f"Saved transcript for {video_id} to {output_file}")

def save_failed_url_to_csv(youtube_url, output_file):
    """
    Append a single failed URL to the no-transcripts CSV file.
    Creates the file with headers if it doesn't exist.
    """
    file_exists = os.path.exists(output_file)
    df = pd.DataFrame([(youtube_url,)], columns=['url'])
    df.to_csv(output_file, index=False, mode='a', header=not file_exists)

def main(csv_file, output_tsv, translate=True, no_transcripts_csv='data/ThumbnailTruthData/no-transcripts.csv', limit_duration=1795):
    """
    Main function to process each YouTube URL, fetch its transcript (limited to 29mins 55 sec), 
    optionally translate, and save each transcript immediately to TSV file.
    If no transcript is available, save the URL to no-transcripts.csv.
    Transcripts are saved incrementally to avoid data loss if API limits are reached.
    """

    df = pd.read_csv(csv_file)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_tsv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load already processed video IDs to skip them (resume capability)
    processed_ids = set()
    if os.path.exists(output_tsv):
        existing_df = pd.read_csv(output_tsv, sep='\t', encoding='utf-8')
        processed_ids = set(existing_df['video_id'].tolist())
        print(f"Found {len(processed_ids)} already processed transcripts, will skip these.")
    
    # Load already failed video IDs to skip them too
    failed_ids = set()
    if os.path.exists(no_transcripts_csv):
        failed_df = pd.read_csv(no_transcripts_csv)
        # Extract video IDs from URLs
        for url in failed_df['url']:
            vid = get_video_id(url)
            if vid:
                failed_ids.add(vid)
        print(f"Found {len(failed_ids)} previously failed videos, will skip these.")
    
    saved_count = 0

    try:
        for youtube_url in df['url']:
            video_id = get_video_id(youtube_url)
            if video_id:
                # Skip if already processed
                if video_id in processed_ids:
                    print(f"Skipping already processed video ID: {video_id}")
                    continue
                
                # Skip if previously failed
                if video_id in failed_ids:
                    print(f"Skipping previously failed video ID: {video_id}")
                    continue
                    
                print(f"Processing video ID: {video_id}")
                time.sleep(22)  # Wait 22 seconds to avoid hitting rate limits

                try:
                    transcript = get_any_transcript(video_id, limit_duration=limit_duration)
                    if transcript:
                        if translate:
                            transcript = translate_transcript(transcript, dest_lang='en')
                        transcript_text = transcript_to_text(transcript)
                        # Save immediately after obtaining each transcript
                        save_transcript_to_tsv(video_id, transcript_text, output_tsv)
                        saved_count += 1
                    else:
                        print(f"No transcript available for video ID: {video_id}")
                        save_failed_url_to_csv(youtube_url, no_transcripts_csv)
                        failed_ids.add(video_id)
                except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, VideoUnplayable, AgeRestricted) as e:
                    print(f"No transcript available for video ID {video_id}: {e}")
                    save_failed_url_to_csv(youtube_url, no_transcripts_csv)
                    failed_ids.add(video_id)
            else:
                print(f"Invalid YouTube URL: {youtube_url}")
                save_failed_url_to_csv(youtube_url, no_transcripts_csv)
                
    except RequestBlocked as e:
        print(f"\n{'='*60}")
        print(f"API RATE LIMIT REACHED!")
        print(f"Error: {e}")
        print(f"Stopping gracefully. {saved_count} transcripts were saved.")
        print(f"Re-run the script later to continue from where you left off.")
        print(f"{'='*60}\n")
    except KeyboardInterrupt:
        print(f"\n{'='*60}")
        print(f"Script interrupted by user.")
        print(f"{saved_count} transcripts were saved before interruption.")
        print(f"Re-run the script to continue from where you left off.")
        print(f"{'='*60}\n")
    
    print(f"\nTotal transcripts saved in this run: {saved_count}")

if __name__ == "__main__":

    thumbnail_type = input("MTV or NMTV?\n")

    if thumbnail_type.lower() == "mtv":
        csv_file = "data/ThumbnailTruthData/mtv.csv"
        output_tsv = "data/ThumbnailTruthData/MTV_transcripts.tsv"
        no_transcripts_csv = "data/ThumbnailTruthData/mtv_no-transcripts.csv"

    elif thumbnail_type.lower() == "nmtv":
        csv_file = "data/ThumbnailTruthData/nmtv.csv"
        output_tsv = "data/ThumbnailTruthData/NMTV_transcripts.tsv"
        no_transcripts_csv = "data/ThumbnailTruthData/nmtv_no-transcripts.csv"
    
    else:
        raise ValueError("Invalid input. Please enter 'MTV' or 'NMTV'.")

    main(csv_file, output_tsv, translate=True, no_transcripts_csv=no_transcripts_csv, limit_duration=1795)

