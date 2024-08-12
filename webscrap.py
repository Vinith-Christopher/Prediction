
# < ---- Import Necessary Packages ---- >
import csv
import bs4
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from termcolor import cprint, colored


def Web_scraping():
    # output file
    outfile = open("BSE_Data.csv", "w", newline='')
    writer = csv.writer(outfile)

    # url tag
    url = "https://finance.yahoo.com/quote/%5EBSESN/history/?guccounter=1&period1=946684800&period2=1723366667"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    # html tags
    soup = bs4.BeautifulSoup(response.text, 'lxml')
    # Table tags
    table_tag = soup.select("table")[0]

    # selecting tr, th, td
    tab_data = [[item.text for item in row_data.select("th,td")]
                for row_data in table_tag.select("tr")]
    # write to the csv file
    for data in tab_data:
        writer.writerow(data)
        # individual rows of the extracted data
        print(' '.join(data))

    cprint(' URL successfully scraped ----', color='green')
