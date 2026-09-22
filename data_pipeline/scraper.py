import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin


# Website address
BASE_URL = "https://books.toscrape.com/"


# This list will store all book details
books = []


# Scrape first 5 catalogue pages
for page_number in range(1, 6):

    # Create page URL
    page_url = urljoin(
        BASE_URL,
        f"catalogue/page-{page_number}.html"
    )

    print("Scraping:", page_url)

    # Get webpage
    response = requests.get(page_url, timeout=10)

    # Check that webpage opened successfully
    response.raise_for_status()

    # Read webpage HTML
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all books on this page
    book_cards = soup.select("article.product_pod")

    print("Books found:", len(book_cards))

    # Read each book
    for book in book_cards:

        # Book title
        title = book.h3.a["title"]

        # Book price
        price = book.select_one(
            ".price_color"
        ).get_text(strip=True)

        # Star rating
        rating = book.select_one(
            "p.star-rating"
        )["class"][1]

        # Availability
        availability = book.select_one(
            ".availability"
        ).get_text(" ", strip=True)

        # Book detail page link
        book_link = book.h3.a["href"]

        detail_url = urljoin(
            page_url,
            book_link
        )

        # Open detail page to get category
        detail_response = requests.get(
            detail_url,
            timeout=10
        )

        detail_response.raise_for_status()

        detail_soup = BeautifulSoup(
            detail_response.text,
            "html.parser"
        )

        # Find category from breadcrumb
        breadcrumb = detail_soup.select_one(
            "ul.breadcrumb"
        )

        breadcrumb_items = breadcrumb.find_all("li")

        category = breadcrumb_items[-2].get_text(
            strip=True
        )

        # Add book information to list
        books.append({
            "title": title,
            "price": price,
            "star_rating": rating,
            "availability": availability,
            "category": category
        })


# Convert the list into a DataFrame
df = pd.DataFrame(books)


# Save the DataFrame as a CSV file
df.to_csv(
    "data_pipeline/books_raw.csv",
    index=False
)


# Show information
print("\nScraping completed!")
print("Total books:", len(df))

print("\nFirst 5 rows:")
print(df.head())

print("\nCSV file created:")
print("data_pipeline/books_raw.csv")