import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import os
import traceback
import sys

sys.stdout.reconfigure(encoding='utf-8')

visited_urls = set()

def crawl_website(url, base_url):
    global visited_urls
    try:
        # Checking if url has already been visited
        if url in visited_urls:
            return None
        
        parsed_url = urlparse(url)
        parsed_based_url = urlparse(base_url)
        if parsed_url.netloc != parsed_based_url.netloc:
            return None
        
        # Create a session
        session = requests.Session()
        
        # Send a GET request to get the URL with additional headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': base_url
        }
        response = session.get(url, headers=headers)
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Add the URL to the set of visited URLs
            visited_urls.add(url)
            
            # Parse the HTML content of the page using Beautiful Soup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text from the page
            extracted_text = soup.get_text()
            
            # Print the URL and extracted text
            print(f"URL: {url}")
            print(extracted_text.strip())
            write_to_file(url, extracted_text.strip())
            
            # Find all internal links on the page and crawl them recursively
            for link in soup.find_all('a', href=True):
                if link is None:
                    continue
                if link['href'].startswith('/'):
                    absolute_url = urljoin(url, link['href'])
                else:
                    absolute_url = link['href']
                crawl_website(absolute_url, base_url)
                
            return extracted_text
        else:
            print(f"Failed to crawl {url}. Status code: {response.status_code}")
            visited_urls.add(url)
            return None
    except Exception as e:
        print(f"An error occured: {e}")
        print(traceback.format_exc())
        visited_urls.add(url)
        return None

def url_to_unique_filename(url):
    # Parse the URL to get the filename
    if url.endswith('/'):
        url = url[:-1]
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    
    # Generate a unique filename using a timestamp
    unique_filename = f"{parsed_url.netloc}_{filename}.txt"
    
    return unique_filename

def write_to_file(url, content):
    os.makedirs('./data_new', exist_ok=True)
    with open('./data_new/' + url_to_unique_filename(url), 'w', encoding='utf-8') as out:
        out.write(content + '\n')
        
def main():
    print('Main thread started')
    
    try:
        startTime = time.time()
        print('Starting crawler')
        url = "https://www.teleperformance.com" # insert url here
        crawl_website(url, url)
        
        print('Run Time', time.time() - startTime)
        
    except Exception as e:
        print(e)
        
if __name__ == '__main__':
    main()