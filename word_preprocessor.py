import re
from tqdm import tqdm

class Word_Preprocessing():
    def __init__(self, df, target):
        self.df = df
        self.target = target

    def eliminate_url(self):
        print('Start eliminate url: : )')
        text = self.df[self.target]
        for i in tqdm(text):
            urls = re.findall(
                r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})',
                i)
            for i in urls:
                self.df[self.target] = self.df[self.target].apply(lambda x: x.replace(i, ""))
        return self.df

    def eliminate_username(self):
        print('Start eliminate username: : )')
        for i in tqdm(self.df[self.target]):
            user_name = re.findall(r'@\w*', i)
            for i in user_name:
                self.df[self.target] = self.df[self.target].apply(lambda x: x.replace(i, ""))
        return self.df

    def convert_abbreviation(self):
        am = "'m"
        are = "'re"
        have = "'ve"
        not_ = "n't"
        self.df[self.target] = self.df[self.target].apply(lambda x: x.replace(am, " am"))
        self.df[self.target] = self.df[self.target].apply(lambda x: x.replace(are, "  are"))
        self.df[self.target] = self.df[self.target].apply(lambda x: x.replace(have, " have"))
        self.df[self.target] = self.df[self.target].apply(lambda x: x.replace(not_, " not"))
        return self.df

    def final_check(self):
        print('Start Final check: ')
        self.df[self.target] = self.df[self.target].apply(
            lambda x: re.sub(r'[^A-Za-z0-9 ]+', ' ', x).lower())
        return self.df

    def eliminate_symbol(self):
        print('Start eliminate symbol: : )')
        symbol_list = [',', "'", '!', '@', '$', '%', '^', '&', '*', '(', ')', '-', '+', '?', '>', '<', '=', '.', ':',
                       ';', '  ', '  ', '   ', '    ', '      ', '      ', '  ']
        for i in tqdm(symbol_list):
            self.df[self.target] = self.df[self.target].apply(lambda x: x.replace(i, ' '))
        return self.df

    def process_all(self):
        self.convert_abbreviation()
        self.eliminate_url()
        self.eliminate_username()
        self.eliminate_symbol()
        df_final_check = self.final_check()
        print("finished!!")
        return df_final_check