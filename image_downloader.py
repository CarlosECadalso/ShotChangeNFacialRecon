from bing_image_downloader import downloader

downloader.download("search", limit=50, output_dir='./downloader_dataset/', adult_filter_off=False)