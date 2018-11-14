import requests, re
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


class FinViz:

    def __init__(self, ticker, important_attributes, min_volume=None, compare_to_markets=False):
        # The ticker for the stock you want to screen
        self.ticker = ticker.upper()
        self.important_attributes = set(important_attributes)

        # Get webpage for ticker you have specified
        self.request = requests.get('https://finviz.com/quote.ashx?t={}&ty=c&ta=1&p=d'.format(self.ticker))
        self.soup = BeautifulSoup(self.request.text, 'html.parser')
        # Need to have the raw values in a list so that it can easily be converted into a np.array
        self.data = {}
        # Create a list of the data values so that it can be converted into an array in numpy
        self.data_values_list = []
        self.data_type_list = []
        self.min_volume = min_volume

        # Run the method on initialisation
        self.get_data()
        self.get_data_values()

        if compare_to_markets:
            self.related_tickers = dict()
            # Find tickers in related sectors
            self.compare_to_market()

    def get_data_values(self):
        """
        Create a list of the data values so that it can be converted into an array in numpy
        """
        data_not_there = set()
        for data_type, data_value in self.data.items():
            if data_type in self.important_attributes:
                is_data_value_number = is_number(data_value)

                if is_data_value_number:
                    self.data_values_list.append(float(data_value))
                    self.data_type_list.append(data_type)
                else:
                    try:
                        self.data_values_list.append(float(data_value[0:len(data_value) - 1]))
                        self.data_type_list.append(data_type)
                    except ValueError:
                        data_not_there.add(data_type)
                        pass
        if data_not_there:
            print(data_not_there)

        assert (len(self.data_type_list) == len(self.data_values_list))

    def get_data(self):
        """
        Get financial data for a given ticker and store the information in a dictionary
        """
        table = self.soup.find('table', attrs={'class': 'snapshot-table2'})
        table_rows = table.find_all('tr')

        for row in table_rows:
            cols = row.find_all('td')
            # 0 - 10 with gaps of 2 is where the relevant information is stored
            for column_number in range(0, 11, 2):
                # Enumerates through first column and next column at the same time to get key and the value to store
                # in a dictionary
                for data_type, data_value in zip(cols[column_number], cols[column_number + 1]):
                    self.data[data_type] = data_value.text.strip()

    def compare_to_market(self):
        """
        Gets all related tickers for the sectors the current ticker is classified in and saves it in a dictionary
        :return:
        """
        # The screener links have a class of tab-link so have to manually look through them all to get it
        all_links = self.soup.find_all(attrs={'class': 'tab-link'})
        # Contains links to get the related companies
        related_links = dict()
        for link in all_links:
            # Get the href value
            href = link.attrs['href']
            # Related companies link has 'screener' in the link
            if 'screener' in href:

                # If we have set a minimum volume, then search for it in the link
                if self.min_volume:
                    # Link getText saves the key as whichever sector it is
                    related_links[link.getText()] = 'https://finviz.com/{},{}'.format(href, self.min_volume)
                else:
                    related_links[link.getText()] = 'https://finviz.com/{}'.format(href)

        for sector, sector_link in related_links.items():
            self.related_tickers[sector] = self.get_related_tickers(related_link=sector_link)

    def get_related_tickers(self, related_link):
        """
        Gets the related tickers for a give sector
        :param related_link: The link for the list of tickers which are relevant in the sector the current ticker is
        classified in
        :return: A list of tickers for the sector in the related_link
        """
        request = requests.get(related_link)
        soup = BeautifulSoup(request.text, 'html.parser')
        div_containing_table = soup.find('div', attrs={'id': 'screener-content'})
        # First table underneath the div
        table = div_containing_table.find('table')

        # Save variable for later set and use
        related_ticker_links = None
        # Where the tickers will be saved
        related_ticker_list = list()
        for index, child in enumerate(table.children):
            if index == 6:
                # All the links for the tickers in this sector
                related_ticker_links = child.find('table').find_all('a', {'class': 'screener-link-primary'})

        for related_link in related_ticker_links:
            # Save the ticker into the list
            related_ticker_list.append(
                FinViz(ticker=related_link.getText(), important_attributes=self.important_attributes))

        return related_ticker_list


class CompareFinViz:

    def __init__(self, fin_viz_objects):
        # fin_viz_objects should be a list of FinViz objects
        self.tickers = fin_viz_objects

    def visualise(self, attribute, sector, main_ticker):
        """
        Plots distribution of whatever attribute is passed in
        :param sector: The sector being visualised and compared against
        :param main_ticker: A FinViz object of the main ticker we are comparing against
        :param attribute: Attribute to be visualised. E.g. PEG.
        """
        # where the data to be plotted will be stored
        x = []

        # Get the data from the objects
        for ticker in self.tickers:
            # Convert value from string to float if it is a a number
            attribute_value = float(ticker.data[attribute]) if is_number(ticker.data[attribute]) else None
            if attribute_value:
                x.append((attribute_value,))
        sns.distplot(x)
        plt.xlabel('{}'.format(attribute))
        plt.ylabel('Density')
        plt.title('{} Plot for all companies in the {} sector'.format(attribute, sector))
        # Draw a line for where the value is for the main ticker we are comparing against
        plt.axvline(x=float(main_ticker.data[attribute]))
        plt.show()


class MachineLearning:

    def __init__(self, fin_viz_objects):
        # A list of FinViz objects, including the main ticker being analysed
        self.fin_viz_objects = fin_viz_objects
        self.x_data = None

    def convert_attributes_to_array(self):
        """
        Converts all the financial attributes of every FinViz instance into a
        """

        self.x_data = np.empty((len(self.fin_viz_objects), len(self.fin_viz_objects[0].data_values_list)))

        for index, instance in enumerate(self.fin_viz_objects):
            self.x_data[index, :] = np.asarray(instance.data_values_list)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def save_ticker_data(finviz_object, ticker):
    # Write information for a ticker to text file
    with open("{}.txt".format(ticker), "w", encoding="utf8") as text_file:
        for key, value in finviz_object.data.items():
            text_file.write('\n {}: {}'.format(key, value))


def setup_volume_options():
    # Setup the volume selection options
    volume_selection = dict()
    # All volume options are represented in 1000s. E.g. The first element 50 signifies a volume of 50,000.
    volume_options = ['50', '100', '200', '300', '400', '500', '750', '1000', '2000']
    for volume in volume_options:
        volume_selection['over_' + volume] = 'sh_avgvol_o' + volume

    return volume_selection


def main():
    volume_selection = setup_volume_options()
    ticker_list = ['MSFT']
    # This list contains any attributes we want learn against. Note that this has already has some attributes removed,
    # such as 52W week range.
    attributes = ['P/E', 'EPS (ttm)', 'Insider Own', 'Shs Outstand', 'Perf Week', 'Market Cap', 'Forward P/E',
                  'EPS next Y',
                  'Insider Trans', 'Shs Float', 'Perf Month', 'Income', 'PEG', 'EPS next Q', 'Short Float',
                  'Perf Quarter', 'Sales', 'P/S', 'EPS this Y', 'Inst Trans', 'Short Ratio', 'Perf Half Y', 'Book/sh',
                  'P/B',
                  'EPS next Y', 'ROA', 'Target Price', 'Perf Year', 'Cash/sh', 'P/C', 'EPS next 5Y', 'ROE', 'Perf YTD',
                  'P/FCF', 'EPS past 5Y', 'ROI', '52W High', 'Beta', 'Quick Ratio',
                  'Sales past 5Y',
                  'Gross Margin', '52W Low', 'ATR', 'Employees', 'Current Ratio', 'Sales Q/Q', 'Oper. Margin',
                  'RSI (14)',
                  'Debt/Eq', 'EPS Q/Q', 'Profit Margin', 'Rel Volume',
                  'LT Debt/Eq', 'Payout', 'Avg Volume', 'Price', 'Recom', 'SMA20', 'SMA50', 'SMA200',
                  'Change']

    # Whether we should save the finiancial data in a txt file
    save_information = False
    visualise_data = False
    attribute_to_visualise = 'PEG'
    compare_to_markets = True
    machine_learning_on = False

    for ticker in ticker_list:
        finviz_object = FinViz(ticker=ticker, min_volume=None,
                               compare_to_markets=compare_to_markets,
                               important_attributes=attributes)

        if compare_to_markets:
            sector = list(finviz_object.related_tickers.keys())[0]
            finviz_objects = finviz_object.related_tickers[sector] + [finviz_object]
        else:
            finviz_objects = [finviz_object]

        if visualise_data:
            sectors = list(finviz_object.related_tickers.keys())
            finviz_objects = finviz_object.related_tickers[sector] + [finviz_object]
            for sector in sectors:
                comparison = CompareFinViz(fin_viz_objects=finviz_objects)
                comparison.visualise(attribute=attribute_to_visualise, sector=sector, main_ticker=finviz_object)

        if save_information:
            save_ticker_data(finviz_object=finviz_object, ticker=ticker)
        # TODO: Go through entire s&p 500 and then get the ones which each company has then do PCA
        if machine_learning_on:
            machine_learning = MachineLearning(fin_viz_objects=finviz_objects)
            machine_learning.convert_attributes_to_array()


if __name__ == "__main__":
    main()
