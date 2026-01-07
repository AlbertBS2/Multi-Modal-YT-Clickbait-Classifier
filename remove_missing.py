import pandas as pd
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


def remove_missing_data(data_csv, no_thumbnail_csv, no_transcripts_csv):
    """
    Remove entries with missing thumbnails or transcripts from the dataset.
    """
    # Load the main dataset
    data = pd.read_csv(data_csv)

    # Load the lists of entries with missing thumbnails and transcripts
    no_thumbnail = pd.read_csv(no_thumbnail_csv)
    no_transcripts = pd.read_csv(no_transcripts_csv)

    # Remove entries with missing thumbnails
    clean_data = data[~data['url'].isin(no_thumbnail['url'])]

    # Remove entries with missing transcripts
    clean_data = clean_data[~clean_data['url'].isin(no_transcripts['url'])]

    # Transform URLs to video IDs
    clean_data['url'] = clean_data['url'].apply(get_video_id)

    # Rename the 'url' column to 'video_id'
    clean_data = clean_data.rename(columns={'url': 'video_id'})

    return clean_data


def test_remove_missing_data(data_csv, no_thumbnail_csv, no_transcripts_csv):
    """
    Check that the number of rows after removing missing data is as expected.
    """
    # Load the main dataset
    data = pd.read_csv(data_csv)

    # Load the lists of entries with missing thumbnails and transcripts
    no_thumbnail = pd.read_csv(no_thumbnail_csv)
    no_transcripts = pd.read_csv(no_transcripts_csv)

    # Remove duplicates in case there are overlapping entries between no_thumbnail and no_transcripts
    no_thumbnail_urls = set(no_thumbnail['url'])
    no_transcripts_urls = set(no_transcripts['url'])
    overlapping_urls = no_thumbnail_urls.intersection(no_transcripts_urls)

    # Calculate expected number of rows after removal
    expected_rows = len(data) - len(no_thumbnail) - len(no_transcripts) + len(overlapping_urls)

    # Remove missing data
    clean_data = remove_missing_data(data_csv, no_thumbnail_csv, no_transcripts_csv)
    assert len(clean_data) == expected_rows, f"Expected {expected_rows} rows, but got {len(clean_data)}"


def main():
    
    thumbnail_type = input("MTV or NMTV?\n")

    if thumbnail_type.lower() == "mtv":
        data_csv = "data/ThumbnailTruthData/mtv.csv"
        no_thumbnail_csv = "data/ThumbnailTruthData/mtv_no-thumb.csv"
        no_transcripts_csv = "data/ThumbnailTruthData/mtv_no-transcripts.csv"
        output_csv = "data/ThumbnailTruthData/mtv_cleaned.csv"

    elif thumbnail_type.lower() == "nmtv":
        data_csv = "data/ThumbnailTruthData/nmtv.csv"
        no_thumbnail_csv = "data/ThumbnailTruthData/nmtv_no-thumb.csv"
        no_transcripts_csv = "data/ThumbnailTruthData/nmtv_no-transcripts.csv"
        output_csv = "data/ThumbnailTruthData/nmtv_cleaned.csv"

    # Run the test
    test_remove_missing_data(data_csv, no_thumbnail_csv, no_transcripts_csv)
    print("Test passed: The number of rows after removal is as expected.")

    # Remove missing data
    clean_data = remove_missing_data(data_csv, no_thumbnail_csv, no_transcripts_csv)

    # Save the cleaned dataset
    clean_data.to_csv(output_csv, index=False)
    print(f"Cleaned data saved to {output_csv}")


if __name__ == "__main__":
    main()