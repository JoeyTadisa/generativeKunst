import collections
import os
import re
import json
import requests
from urllib.parse import urlparse, urljoin
from urllib.request import urlretrieve
from requests.exceptions import RequestException
# from unidecode import unidecode
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from PIL import Image
from io import BytesIO

# def extract_collections_label(soup, ):
#     """
#     <label data-hook="label" 
#     class="wixSdkShowFocusOnSibling o8pQoW">
#     <input data-hook="radio-option-input" 
#     type="radio" 
#     class="r36wK7"
#     tabindex="-1"
#     value="6da46704-2fb9-1279-3726-ea82a801007e">
#     Curves
#     </label>
#     """

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_categories(text):
    """
    Extracts categories and their associated values from a given text.

    Args:
        text (str): The input text containing categories and values.

    Returns:
        dict: A dictionary with category names as keys and their associated values as values. Returns None for missing categories.

    Example:
        input_text = "Maße: 80 x80 cmMedium: Acryl auf LeinwandTechnik: Pinsel, SpachtelEntstehungsjahr: 2023 Seit vielen Jahren gehört Meditation zu meiner täglichen Morgenroutine ..."
        result = extract_categories(input_text)
        print(result)
    """
    # Define the keywords and their respective regex patterns
    keywords = {
        "Maße": r"Maße:\s*([^M]+)",
        "Medium": r"Medium:\s*([^T]+)",
        "Technik": r"Technik:\s*([^E]+)",
        "Entstehungsjahr": r"Entstehungsjahr:\s*([\d]+)"
    }

    # Initialize a dictionary to store extracted values
    extracted_values = {}

    # Process each category one by one
    for keyword, pattern in keywords.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            extracted_values[keyword] = value
            text = text[:match.start()] + text[match.end():]
        else:
            extracted_values[keyword] = None

    # The remaining text is considered as captions
    extracted_values["Captions"] = text.strip()

    return extracted_values

def extract_metadata_element(soup, property_name):
    """
    Extract a metadata element from a BeautifulSoup object.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object of the HTML page.
        property_name (str): The name of the metadata property.

    Returns:
        str: The content of the metadata element or an empty string if not found.
    """
    tag = soup.find('meta', attrs={'property': property_name})
    return tag.get('content') if tag else ''

def download_and_save_image(image_url, image_filename):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        with open(image_filename, 'wb') as image_file:
            image_file.write(response.content)
    except RequestException as e:
        print(f"Error downloading image from {image_url}: {e}")
        
def extracting_product_urls(href):
    """
    Extracts product URLs with pagination from a web page and stores them in a CSV file.

    Parameters:
    href (str): The URL of the web page to extract product URLs from.

    Global Variables:
    csv_file_created (bool): Tracks whether the CSV file has been created.

    This function processes a web page with pagination to extract product URLs by
    locating 'a' elements with the class 'JPDEZd'. It iterates through multiple pages
    by incrementing the 'page' counter and constructs URLs with pagination. The
    matching URLs are collected and either saved to a CSV file ('matching_links.csv')
    or appended to an existing CSV file without a header if it already exists.

    Note:
    The global variable 'csv_file_created' is used to determine whether the CSV file
    already exists to avoid overwriting headers.

    """
    print("Processing:", href)

    try:
        # Initialize a list to store all matching links
        matching_links = []

        # Initialize page counter for pagination
        page = 1

        while True:
            # Construct the URL with pagination
            if page == 1:
                page_url = href
            else:
                page_url = f"{href}?page={page}"

            # Fetch the web page content using requests
            response = requests.get(page_url)
            page_source = response.text

            # Parse the page source with BeautifulSoup
            soup = BeautifulSoup(page_source, 'html.parser')

            # Find all 'a' elements with class 'JPDEZd'
            matching_a_tags = soup.find_all('a', class_='JPDEZd')

            if not matching_a_tags:
                # No more matching links found, break the loop
                break

            # Extract the href attribute from matching links and append to matching_links
            matching_links.extend([a['href'] for a in matching_a_tags])

            # Increment the page counter
            page += 1

        # Process the matching links or save them to a CSV file
        if matching_links:
            global csv_file_created
            if not csv_file_created:
                # Create the CSV file with the header
                df = pd.DataFrame(matching_links, columns=['Matching Links'])
                df.to_csv('matching_links.csv', index=False)
                csv_file_created = True
            else:
                # Append the data to the existing CSV file without the header
                df = pd.DataFrame(matching_links, columns=['Matching Links'])
                df.to_csv('matching_links.csv', mode='a', index=False, header=False)

    except Exception as e:
        # Handle exceptions in a way that won't stop the program
        print(f"Exception occurred for {href}: {str(e)}")

def extract_metadata(url):
    """
    Extract metadata from a given URL using BeautifulSoup and requests.

    Parameters:
        url (str): The URL from which to extract metadata.

    Returns:
        dict: A dictionary containing the extracted metadata.

    The function sends a GET request to the specified URL, retrieves the HTML content, and then
    parses it with BeautifulSoup. It extracts various metadata elements from the page and returns
    them as a dictionary.

    Example usage:
    url = "https://www.example.com"
    metadata = extract_metadata(url)
    if metadata:
        print(metadata)
    """

    # Send a GET request to the URL and retrieve the content
    response = requests.get(url)
    if response.status_code != 200:
        return None

    page_content = response.text
    soup = BeautifulSoup(page_content, 'html.parser')

    # Extract metadata elements using the generic function
    metadata = {
        "URL": url,
        "og_title": extract_metadata_element(soup, 'og:title'),
        "og_image": extract_metadata_element(soup, 'og:image'),
        "og_image_width": extract_metadata_element(soup, 'og:image:width'),
        "og_image_height": extract_metadata_element(soup, 'og:image:height')
    }

    # Extract categories and captions from meta description
    meta_description = extract_metadata_element(soup, 'og:description')
    result = extract_categories(meta_description)

    metadata.update({
        "Maße": result["Maße"],
        "Medium": result["Medium"],
        "Technik": result["Technik"],
        "Entstehungsjahr": result["Entstehungsjahr"],
        "meta_description": result["Captions"]
    })

    return metadata

# Function to check if an image meets the resolution criteria
def is_resolution_above(image_url, width_threshold, height_threshold):
    try:
        response = requests.head(image_url)
        if response.status_code == 200:
            content_length = int(response.headers['Content-Length'])
            if content_length > width_threshold * height_threshold:
                return True
    except Exception as e:
        print(f"ERROR: {e}")
        pass
    return False

def check_duplicate_titles(meta):
    titles = meta['og_title'].tolist()
    duplicates = [item for item, count in collections.Counter(titles).items() if count > 1]
    if duplicates:
        print(f"Warning: Found duplicate titles: {duplicates}")
    else:
        print("No duplicate titles found.")
        
def verify_image(image_path):
    try:
        img = Image.open(image_path) # open the image file
        img.verify() # verify that it is, in fact, an image
        return True
    except (IOError, SyntaxError) as e:
        print('Bad file:', image_path)
        return False

# # vars
# base_url = 'https://www.christin-kirchner.com/abstraktekunstwerke'
# img_sources = list()
# img_src_html_class = "JPDEZd" # hyperlink class/a-tag

# # Minimum resolution criteria
# width_threshold = 500
# height_threshold = 500

# # Send an HTTP GET request to the URL
# response = requests.get(base_url)

# # Parse the HTML content of the page
# soup = BeautifulSoup(response.text, 'html.parser')

# # Find all image tags in the HTML
# img_tags = soup.find_all('img')

# # Loop through each image tag, check resolution, and download the image
# for img_tag in img_tags:
#     img_url = img_tag.get('src')
    
#     # Make sure the URL is valid
#     if img_url:
#         img_url = urljoin(base_url, img_url)
#         if is_resolution_above(img_url, width_threshold, height_threshold):
#             download_and_save_image(img_url, kunst_dump)

# Load the list of URLs to extract metadata from
meta = pd.read_csv('kunst_metadata.csv')
check_duplicate_titles(meta)

# Create an empty DataFrame to store the metadata
metadata_df = pd.DataFrame()
kunst_metadata = [] # Initialize a list to store metadata for the .jsonl file
kunst_dump = "./data/full-finetune/all_kunst_v2/" # contains new default and title is used in caption as well.
os.makedirs(kunst_dump, exist_ok=True)

successful_downloads = list()  # List to store the titles of successfully downloaded images

# Assuming meta is your DataFrame
for index, row in tqdm(meta.iterrows(), total=len(meta)):
    # Download and save the image
    og_title = row['og_title']
    image_url = row['og_image']
    if image_url:
        # Convert title to ASCII
        og_title = og_title.replace('|', '')
        og_title = og_title.replace('verkauft', '')
        image_filename = f'{kunst_dump}{og_title}.png'
        try:
            download_and_save_image(image_url, image_filename)
            successful_downloads.append(og_title, )  # Add the title to the list of successful downloads
            text_value = row['meta_description'] if pd.notna(row['meta_description']) else "Abstract Artwork by Christin Kirchner"
            text_value = text_value.replace('verkauft', '')
            metadata_entry = {
                "file_name": f"{og_title}.png",
                "text": f"{og_title}{text_value}"
            }
            kunst_metadata.append(metadata_entry)
        except Exception as e:
            print(f"Failed to download image from {image_url}. Error: {e}")
            
print(len(kunst_metadata))
print(kunst_metadata)
meta_dump = list(set(tuple(sorted(d.items())) for d in kunst_metadata))
meta_dump = [dict(t) for t in meta_dump]

# Save the metadata to a .jsonl file
jsonl_file_path = os.path.join(kunst_dump, "metadata.jsonl")
with open(jsonl_file_path, 'w', encoding='utf-8') as jsonl_file:
    for meta in meta_dump:
        jsonl_file.write(json.dumps(meta) + '\n')

print('Done')

# import os
# import re
# import json
# import requests
# import pandas as pd
# from bs4 import BeautifulSoup
# from tqdm import tqdm
# from urllib.parse import urljoin, quote_plus
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.options import Options as ChromeOptions
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.chrome.service import Service as ChromiumService
# from webdriver_manager.chrome import ChromeDriverManager
# from webdriver_manager.core.os_manager import ChromeType
# from selenium.webdriver.support import expected_conditions as EC

# def create_directory(directory):
#     if not os.path.exists(directory):
#         os.makedirs(directory)

# def download_and_save_image(image_url, image_filename):
#     try:
#         response = requests.get(image_url)
#         response.raise_for_status()
#         with open(image_filename, 'wb') as image_file:
#             image_file.write(response.content)
#     except requests.RequestException as e:
#         print(f"Error downloading image from {image_url}: {e}")

# def extract_metadata(url):
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         soup = BeautifulSoup(response.text, 'html.parser')

#         metadata = {
#             "URL": url,
#             "og_title": extract_metadata_element(soup, 'og:title'),
#             "og_image": extract_metadata_element(soup, 'og:image'),
#             "og_image_width": extract_metadata_element(soup, 'og:image:width'),
#             "og_image_height": extract_metadata_element(soup, 'og:image:height')
#         }

#         meta_description = extract_metadata_element(soup, 'og:description')
#         result = extract_categories(meta_description)

#         metadata.update({
#             "Maße": result["Maße"],
#             "Medium": result["Medium"],
#             "Technik": result["Technik"],
#             "Entstehungsjahr": result["Entstehungsjahr"],
#             "meta_description": result["Captions"]
#         })

#         return metadata
#     except requests.RequestException as e:
#         print(f"Error extracting metadata from {url}: {e}")
#         return None

# def extract_categories(text):
#     keywords = {
#         "Maße": r"Maße:\s*([^M]+)",
#         "Medium": r"Medium:\s*([^T]+)",
#         "Technik": r"Technik:\s*([^E]+)",
#         "Entstehungsjahr": r"Entstehungsjahr:\s*([\d]+)"
#     }

#     extracted_values = {}

#     for keyword, pattern in keywords.items():
#         match = re.search(pattern, text)
#         if match:
#             value = match.group(1).strip()
#             extracted_values[keyword] = value
#             text = text[:match.start()] + text[match.end():]
#         else:
#             extracted_values[keyword] = None

#     extracted_values["Captions"] = text.strip()

#     return extracted_values

# def extract_metadata_element(soup, property_name):
#     tag = soup.find('meta', attrs={'property': property_name})
#     return tag.get('content', '')

# def is_resolution_above(image_url, width_threshold, height_threshold):
#     try:
#         response = requests.head(image_url)
#         if response.status_code == 200:
#             content_length = int(response.headers.get('Content-Length', 0))
#             return content_length > width_threshold * height_threshold
#     except requests.RequestException as e:
#         print(f"ERROR: {e}")
#     return False

# def extract_image_labels(soup):
#     labels = []
#     parent_elements = soup.find_all('li', class_='hYcCbd')
#     for parent_element in parent_elements:
#         # Extract label from the parent element
#         label = parent_element.find('label').text.strip()
#         # Disregard "Alle" (customize this condition based on your needs)
#         if label.lower() != 'alle':
#             labels.append(label)
#     return labels

# def generate_collection_urls(base_url, labels):
#     collection_urls = []
#     for label in labels:
#         # Encode the label for the URL
#         encoded_label = quote_plus(label)
#         # Generate the collection URL using the structure
#         collection_url = f"{base_url}?Kategorie={encoded_label}"
#         collection_urls.append(collection_url)
#     return collection_urls

# def save_to_text_file(file_path, data):
#     with open(file_path, 'w') as file:
#         for item in data:
#             file.write(f"{item}\n")
            
# def extracting_product_urls(href):
#     """
#     Extracts product URLs with pagination from a web page and stores them in a CSV file.

#     Parameters:
#     href (str): The URL of the web page to extract product URLs from.

#     Global Variables:
#     csv_file_created (bool): Tracks whether the CSV file has been created.

#     This function processes a web page with pagination to extract product URLs by
#     locating 'a' elements with the class 'JPDEZd'. It iterates through multiple pages
#     by incrementing the 'page' counter and constructs URLs with pagination. The
#     matching URLs are collected and either saved to a CSV file ('matching_links.csv')
#     or appended to an existing CSV file without a header if it already exists.

#     Note:
#     The global variable 'csv_file_created' is used to determine whether the CSV file
#     already exists to avoid overwriting headers.

#     """
#     print("Processing:", href)

#     try:
#         # Initialize a list to store all matching links
#         matching_links = []

#         # Initialize page counter for pagination
#         page = 1

#         while True:
#             # Construct the URL with pagination
#             if page == 1:
#                 page_url = href
#             else:
#                 page_url = f"{href}?page={page}"

#             # Fetch the web page content using requests
#             response = requests.get(page_url)
#             page_source = response.text

#             # Parse the page source with BeautifulSoup
#             soup = BeautifulSoup(page_source, 'html.parser')

#             # Find all 'a' elements with class 'JPDEZd'
#             matching_a_tags = soup.find_all('a', class_='JPDEZd')

#             if not matching_a_tags:
#                 # No more matching links found, break the loop
#                 break

#             # Extract the href attribute from matching links and append to matching_links
#             matching_links.extend([a['href'] for a in matching_a_tags])

#             # Increment the page counter
#             page += 1

#         # Process the matching links or save them to a CSV file
#         if matching_links:
#             global csv_file_created
#             if not csv_file_created:
#                 # Create the CSV file with the header
#                 df = pd.DataFrame(matching_links, columns=['Matching Links'])
#                 df.to_csv('matching_links.csv', index=False)
#                 csv_file_created = True
#             else:
#                 # Append the data to the existing CSV file without the header
#                 df = pd.DataFrame(matching_links, columns=['Matching Links'])
#                 df.to_csv('matching_links.csv', mode='a', index=False, header=False)

#     except Exception as e:
#         # Handle exceptions in a way that won't stop the program
#         print(f"Exception occurred for {href}: {str(e)}")
            
# def collect_images_by_collection(collection_url, label, width_threshold=500, height_threshold=500, gather_collections=False):
#     output_dir = "./data/full-finetune/"
#     dump_all = "all_kunst"

#     # Determine the dump directory based on the toggle
#     if gather_collections:
#         collection_dump = os.path.join(output_dir, dump_all)
#     else:
#         collection_dump = os.path.join(output_dir, label)
#         os.makedirs(collection_dump, exist_ok=True)
#         print(f"Directory created for collection {label}")

#     print("Processing:", collection_url)

#     driver = None  # Initialize driver as None

#     try:
#         product_pages = []
#         current_page_count = 0

#         # Set up ChromeOptions to use --no-sandbox
#         chrome_options = ChromeOptions()
#         chrome_options.add_argument('--no-sandbox')

#         # Use Selenium to interact with the dynamic content
#         driver = webdriver.Chrome(service=ChromiumService(ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install()), options=chrome_options)
#         driver.get(collection_url)

#         while True:
#             # Wait for the "Mehr laden" button to be clickable
#             mehr_laden_button = WebDriverWait(driver, 10).until(
#                 EC.element_to_be_clickable((By.CLASS_NAME, 'txtqbB'))
#             )

#             # Click the "Mehr laden" button
#             mehr_laden_button.click()

#             # Get the updated page source
#             page_source = driver.page_source
#             soup = BeautifulSoup(page_source, 'html.parser')

#             # Find all 'a' elements with class 'JPDEZd'
#             matching_a_tags = soup.find_all('a', class_='JPDEZd')
#             new_page_count = len(matching_a_tags)

#             if new_page_count == current_page_count:
#                 # No more matching links found, break the loop
#                 break

#             # Extract the href attribute from matching links and append to matching_links
#             product_pages.extend([a['href'] for a in matching_a_tags])
#             print(product_pages)
#             current_page_count = new_page_count
#     except Exception as e:
#         print(f"Exception occurred while handling {collection_url}: {str(e)}")
#     finally:
#         if driver:
#             # Close the Selenium WebDriver if it has been created
#             driver.quit()

#     return product_pages
# def main():
#     base_url = 'https://www.christin-kirchner.com/abstraktekunstwerke'
#     base_output_path = "./data/full-finetune/"
#     kunst_dump = "./data/full-finetune/all_kunst_playground/"
#     collections_dump = './data/full-finetune/collection_urls.txt'
#     csv_file_path = 'kunst_metadata.csv'
    
#     os.makedirs(kunst_dump, exist_ok=True)

#     # Minimum resolution criteria
#     width_threshold = 500
#     height_threshold = 500

#     # Send an HTTP GET request to the URL
#     response = requests.get(base_url)
#     soup = BeautifulSoup(response.text, 'html.parser')

#     # Extract image labels
#     collection_labels = extract_image_labels(soup)
#     print(collection_labels)

#     # Generate collection URLs
#     collection_urls = generate_collection_urls(base_url, collection_labels)

#     # Save collection URLs to a text file
#     save_to_text_file(collections_dump, collection_urls)

#     print('Collection URLs saved to:', collections_dump)
    
#     for url, label in zip(collection_urls, collection_labels):
#         product_page_urls = collect_images_by_collection(url, label, 500, 500, False)
        
#     print(product_page_urls)

    # # Find all image tags in the HTML
    # img_tags = soup.find_all('img')

    # # Loop through each image tag, check resolution, and download the image
    # for img_tag, label_info in zip(img_tags, image_labels):
    #     img_url = img_tag.get('src')

    #     # Make sure the URL is valid
    #     if img_url:
    #         img_url = urljoin(base_url, img_url)
    #         if is_resolution_above(img_url, width_threshold, height_threshold):
    #             og_title = os.path.splitext(os.path.basename(img_url))[0]
    #             image_filename = os.path.join(kunst_dump, f'{og_title}.jpg')
    #             download_and_save_image(img_url, image_filename)

    #             # Use label_info in your logic to update the kunst_metadata.csv file
    #             # label_info contains {"label": "Love Letters", "value": "2de691ef-e5e3-8e4a-8429-ef43c2aad31b"}

    # # Load the list of URLs to extract metadata from
    # meta = pd.read_csv(csv_file_path)

    # # Create an empty DataFrame to store the metadata
    # metadata_df = pd.DataFrame()
    # kunst_metadata = []  # Initialize a list to store metadata for the .jsonl file

    # for index, row in tqdm(meta.iterrows(), total=len(meta)):
    #     # Download and save the image
    #     og_title = row['og_title']
    #     image_url = row['og_image']
    #     if image_url:
    #         image_filename = f'{kunst_dump}{og_title}.jpg'
    #         download_and_save_image(image_url, image_filename)

    #         # Create metadata entry for the .jsonl file
    #         text_value = row['meta_description'] if pd.notna(row['meta_description']) else ""
    #         metadata_entry = {
    #             "file_name": f"{og_title}.jpg",
    #             "text": text_value
    #         }

    #         kunst_metadata.append(metadata_entry)

    # # Save the metadata to a .jsonl file
    # jsonl_file_path = os.path.join(kunst_dump, "metadata.jsonl")
    # with open(jsonl_file_path, 'w') as jsonl_file:
    #     for entry in kunst_metadata:
    #         jsonl_file.write(json.dumps(entry) + '\n')

    # print('Done')

# if __name__ == "__main__":
#     main()
