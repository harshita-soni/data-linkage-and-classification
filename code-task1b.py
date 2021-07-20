import pandas as pd
from fuzzywuzzy import process
import re


def main():
    abt_df = pd.read_csv('abt.csv', encoding='ISO-8859-1')
    buy_df = pd.read_csv('buy.csv', encoding='ISO-8859-1')

    brands_all = buy_df['manufacturer'].unique()
    brands = set([str(brand).lower().split()[0] for brand in brands_all if str(brand) != 'nan'])

    blocks_buy = {'block_key': [], 'product_id': []}
    for index, record in buy_df.iterrows():
        if str(record['manufacturer']) == 'nan':
            for word in record['name'].lower().split():
                if re.search(r'\d', word) is None and word.isalpha():
                    blocks_buy['block_key'].append(word)
                    blocks_buy['product_id'].append(record['idBuy'])
                    break
        else:
            blocks_buy['block_key'].append(record['manufacturer'].lower().split()[0])
            blocks_buy['product_id'].append(record['idBuy'])

    blocks_buy_df = pd.DataFrame.from_dict(blocks_buy)
    blocks_buy_df.to_csv('buy_blocks.csv', index=False)

    blocks_abt = {'block_key': [], 'product_id': []}
    for index, record in abt_df.iterrows():
        blocks_abt['block_key'].append(get_manufacturer(record['name'], brands))
        blocks_abt['product_id'].append(record['idABT'])

    blocks_abt_df = pd.DataFrame.from_dict(blocks_abt)
    blocks_abt_df.to_csv('abt_blocks.csv', index=False)


def get_manufacturer(name, brands):
    name = ' '.join([word for word in name.lower().split() if word != "-"][:2])
    for brand in brands:
        if name.split()[0] == str(brand):
            return brand
    #
    # ratios = process.extract(name, brands)
    # sorted_ratios = sorted(ratios, key=lambda x: x[1], reverse=True)
    # brand, ratio = sorted_ratios[0]
    # if ratio > 50:
    #     return brand
    # return ''


if __name__ == "__main__":
    main()
