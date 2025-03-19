from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import json


def setup_linkedin_scraper():
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Initialize the Chrome driver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    return driver


def collect_linkedin_posts(driver, search_terms, max_posts=100):
    collected_posts = []

    # Login logic would be placed here
    # Note: Actual implementation would need proper authentication

    for term in search_terms:
        # Navigate to search results
        driver.get(f"https://www.linkedin.com/search/results/content/?keywords={term}")
        time.sleep(3)  # Allow page to load

        # Scroll to load more content
        for _ in range(5):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

        # Extract post content
        posts = driver.find_elements(By.XPATH, "//div[contains(@class, 'feed-shared-update-v2')]")

        for post in posts[:max_posts]:
            try:
                content = post.find_element(By.XPATH, ".//div[contains(@class, 'feed-shared-text')]").text
                author = post.find_element(By.XPATH, ".//span[contains(@class, 'feed-shared-actor__name')]").text

                post_data = {
                    "content": content,
                    "author": author,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }

                collected_posts.append(post_data)
            except Exception as e:
                print(f"Error extracting post: {e}")

    driver.quit()
    return collected_posts
