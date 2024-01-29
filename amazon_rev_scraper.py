from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import csv
import time

# Setup WebDriver
gecko_driver_path = "C:\WebDriver\geckodriver\geckodriver.exe"  # Update this path
s = Service(gecko_driver_path)
driver = webdriver.Firefox(service=s)

# Function to get reviews from a single page
def get_reviews(driver):
    reviews = []
    review_elements = driver.find_elements(By.XPATH, '//div[@data-hook="review"]')
    for review_element in review_elements:
        try:
            title = review_element.find_element(By.XPATH, './/a[@data-hook="review-title"]/span').text
            body = review_element.find_element(By.XPATH, './/span[@data-hook="review-body"]/span').text

            # Replace newline characters with a space
            title = title.replace('\n', ' ')
            body = body.replace('\n', ' ')

            reviews.append({'title': title, 'body': body})
        except NoSuchElementException:
            continue
    return reviews


# Open the Amazon login page
driver.get("https://www.amazon.com/gp/sign-in.html")

# Wait for manual login
input("Log in to Amazon and then press Enter in this terminal to continue...")

# Navigate to the product reviews page after logging in
product_url = "https://www.amazon.com/LEVOIT-Purifiers-Bedroom-Washable-Vital100S/product-reviews/B0BNDM2RNG/ref=cm_cr_getr_d_paging_btm_prev_1?ie=UTF8&reviewerType=all_reviews&pageNumber=1"
driver.get(product_url)

all_reviews = []

# Scrape multiple pages of reviews
while True:
    all_reviews.extend(get_reviews(driver))
    try:
        next_button = driver.find_element(By.XPATH, '//li[@class="a-last"]/a')
        if "a-disabled" in next_button.get_attribute("class"):
            break
        next_button.click()
        time.sleep(2)  # Adjust as needed
    except NoSuchElementException:
        break  # No more pages of reviews

driver.quit()

# Save the reviews to a CSV file
csv_file_path = 'D:/EDUCATION_MID_LIFE/UNIBO_MS_Analytics/Y2Sem1/WebnSocialMining/amazon_reviews.csv'
with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=['title', 'body'])
    writer.writeheader()
    for review in all_reviews:
        writer.writerow(review)

print(f"Saved {len(all_reviews)} reviews to {csv_file_path}")
