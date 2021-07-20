import pandas as pd
from textdistance import cosine


def main():
    abt = pd.read_csv('abt_small.csv', encoding='ISO-8859-1')
    buy = pd.read_csv('buy_small.csv', encoding='ISO-8859-1')
    matched = {'idAbt': [], 'idBuy': []}

    for index_a, record_a in abt.iterrows():
        for index_b, record_b in buy.iterrows():
            similar = are_similar(record_a['name'].lower(), record_b['name'].lower(),
                                  record_b['manufacturer'].lower().split()[0],
                                  get_model_number(record_a['name']), get_model_number(record_b['name']))
            # match solely on the basis of model numbers and  manufacturer names
            if similar:
                matched['idAbt'].append(record_a['idABT'])
                matched['idBuy'].append(record_b['idBuy'])
                break

    for index_a, record_a in abt.iterrows():
        maxsim = 0.0
        # now for each record that couldn't be matched earlier, use cosine similarity to possible find matches
        for index_b, record_b in buy.iterrows():
            sim = cosine(record_a['name'].lower(), record_b['name'].lower())

            if (record_b['idBuy'] not in matched['idBuy'] and record_a['idABT'] not in matched['idAbt']
                    and sim > maxsim):
                model1 = get_model_number(record_a['name'])
                model2 = get_model_number(record_b['name'])

                if (brand_same(record_a['name'].lower(), record_b['name'].lower(),
                               record_b['manufacturer'].lower().split()[0])) and sim >= 0.6:
                    maxsim = sim
                    match = (record_a['idABT'], record_b['idBuy'])

                elif (model1 and model2) and not models_same(model1, model2) and sim >= 0.6:
                    maxsim = sim
                    match = (record_a['idABT'], record_b['idBuy'])

                elif not model1 and not model2 and sim >= 0.6:
                    maxsim = sim
                    match = (record_a['idABT'], record_b['idBuy'])

        if maxsim >= 0.80 and match:
            matched['idAbt'].append(match[0])
            matched['idBuy'].append(match[1])

    matching_pairs = pd.DataFrame.from_dict(matched)
    matching_pairs.to_csv('task1a.csv', index=False)


def are_similar(s1, s2, brand2, model1, model2):
    if not brand_same(s1, s2, brand2):
        return False

    if models_same(model1, model2):
        return True


def brand_same(s1, s2, brand2):
    brand1 = s1.lower().split()[0]
    if not (brand1 in brand2 or brand2 in brand1 or
            brand1 in s2.lower().split()[0] or s2.lower().split()[0] in brand1):
        return False
    return True


def get_model_number(name):
    model_numbers = list()
    for word in name.split():
        if ',' in word or '.' in word:
            continue
        if (len(word) >= 4) and (word.upper() == word):
            for char in word:
                if char.isdigit():
                    word = word.replace('-', '')
                    word = word.replace('/', '')
                    model_numbers.append(word)
    return set(model_numbers)


def models_same(model1, model2):
    if model1 and model2:
        for m in model1:
            for n in model2:
                if m in n or n in m:
                    return True
    return False


if __name__ == "__main__":
    main()
